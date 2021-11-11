import numpy as np
import torch
import os
import copy
import time
import gym
import psutil
from wrappers import PointCloudWrapper

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageDraw

from collections import defaultdict


def makeEnv(env_name, idx, args):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        e = gym.make('{}-rotate-v1'.format(env_name))
        e.seed(args.seed + idx)
        return PointCloudWrapper(e, args)
    return helper


def get_avg_values_across_batches(values):
    """
    values: list(dict) where each dict is the infomration returned by _update_network()
    """
    # convert list(dict) to dict(list)
    all_keys = []
    for v in values:
        all_keys += list(v.keys())
    all_keys = set(all_keys)
    values = {k: [dic[k] for dic in values if k in dic.keys()]
              for k in all_keys}
    mean_across_batch = dict()
    for k in values.keys():
        if isinstance(values[k][0], torch.Tensor):
            mean_across_batch[k] = torch.mean(
                torch.stack(values[k])).detach().cpu().numpy()
        else:
            mean_across_batch[k] = np.mean(values[k])
    return mean_across_batch


def get_memory_usage():
    process = psutil.Process(os.getpid())
    megabytes = process.memory_info().rss / 1024 ** 2
    return megabytes


def dictArray2arrayDict(x):
    """convert an array of dictionary to dictionary of arrays"""
    assert isinstance(x[0], dict)
    keys = list(x[0].keys())
    res = dict()
    for k in x[0].keys():
        res[k] = np.array([e[k] for e in x])
    return res


def addSTARTtoImages(imgs):
    for i in range(len(imgs)):
        img = imgs[i]
        img = Image.fromarray(img, "RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('sans-serif.ttf', 60)
        draw.text((170, 380), "START", (0, 0, 0), font=font)
        img = np.array(img)
        imgs[i] = img.astype(np.uint8)
    return imgs


def addSUCCESStoImages(imgs, filter_array):
    for i in range(len(imgs)):
        if filter_array[i]:
            img = imgs[i]
            img = Image.fromarray(img, "RGB")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('sans-serif.ttf', 60)
            draw.text((115, 380), "SUCCESS", (0, 0, 0), font=font)
            img = np.array(img, dtype=np.uint8)
            mask = (img == 255).astype(np.uint8)
            img = mask * np.array([[[183, 230, 165]]],
                                  dtype=img.dtype) + (1 - mask) * img
            imgs[i] = img.astype(np.uint8)
    return imgs


def preproc_og(o, g, clip_obs=200):
    o = np.clip(o, -clip_obs, clip_obs)
    g = np.clip(g, -clip_obs, clip_obs)
    return o, g


def preproc_inputs(obs, g, o_normalizer, g_normalizer, cuda=False):
    device = torch.device("cuda" if cuda else "cpu")
    obs_norm = o_normalizer.normalize(obs)
    g_norm = g_normalizer.normalize(g)
    # concatenate the stuffs
    inputs = np.concatenate([obs_norm, g_norm], axis=-1)
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    if len(inputs.shape) == 1:
        inputs = inputs.unsqueeze(0)
    return inputs


def distance_between_rotations(q1, q2):
    """
    calculate distance between two unit quaternions
    from the following paper: 
                                    Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning
                                    (https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf)
    """
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4, 'must be quaternions'
    return 1 - (np.sum(q1 * q2, axis=-1)) ** 2


def collect_experiences(env, num_episodes, max_timesteps, actor, o_normalizer, g_normalizer, env_params,
                        cuda=False, action_proc_func=None, point_cloud=False,
                        video_count=0, vec_env_names=None):
    # define helper attributes for vectorized envs =================================================================
    num_envs = env.num_envs  # number of parallel vec envs
    assert len(vec_env_names) % len(set(vec_env_names)) == 0
    total_success_rate, total_distance, total_reward, total_video = [], [], [], []
    assert num_episodes != 0
    assert num_episodes % num_envs == 0, 'num_episodes ({}) must be multiple of num of parallel envs ({})'.format(
        num_episodes, num_envs)
    obs_key = 'pc_obs' if point_cloud else 'minimal_obs'
    # assert video_count <= 1
    # define storage =================================================================
    mb_obs = np.empty([num_episodes, max_timesteps + 1, env_params['obs']])
    mb_ag = np.empty([num_episodes, max_timesteps + 1, env_params['goal']])
    mb_g = np.empty([num_episodes, max_timesteps, env_params['goal']])
    mb_actions = np.empty([num_episodes, max_timesteps, env_params['action']])
    mb_r = np.empty([num_episodes, max_timesteps, 1])
    # start collection =================================================================
    episodes_collected = 0
    while episodes_collected < num_episodes:
        per_reward = np.zeros(num_envs)
        per_video = []
        # reset the environment
        observation = env.reset()
        if not isinstance(observation, dict):
            observation = dictArray2arrayDict(observation)
        # iterate through rollouts =================================================================
        for t in range(max_timesteps):
            with torch.no_grad():
                input_tensor = preproc_inputs(
                    np.array(observation[obs_key]), observation['desired_goal'], o_normalizer, g_normalizer, cuda)
                assert len(input_tensor.shape) == 2
                pi = actor(input_tensor)
                # add noise to action output for exploration if needed
                if action_proc_func is None:
                    action = pi.detach()
                else:
                    action = np.empty(pi.shape, dtype=np.float32)
                    for idx in range(pi.shape[0]):
                        action[idx, :] = action_proc_func(pi[idx, :])
            # feed the actions into the environment
            observation_new, reward, done, info = env.step(action if isinstance(
                action, np.ndarray) else action.cpu().numpy().squeeze())
            if not isinstance(observation_new, dict):
                observation_new = dictArray2arrayDict(observation_new)
            # last step of vec env would in fact be the first step of next env (automatic reset); so store ahead
            if t == max_timesteps - 2:
                final_observation = copy.deepcopy(observation_new)
            # store rollout data =====================================================================================
            mb_obs[episodes_collected:(
                episodes_collected + num_envs), t, :] = np.array(observation[obs_key]).copy()
            mb_ag[episodes_collected:(
                episodes_collected + num_envs), t, :] = np.array(observation['achieved_goal']).copy()
            mb_g[episodes_collected:(
                episodes_collected + num_envs), t, :] = np.array(observation['desired_goal']).copy()
            mb_r[episodes_collected:(
                episodes_collected + num_envs), t, :] = reward[..., None].copy()
            mb_actions[episodes_collected:(episodes_collected + num_envs), t, :] = action.squeeze(
            ) if isinstance(action, np.ndarray) else action.cpu().numpy().squeeze()
            per_reward += reward
            # store rollout data =====================================================================================
            if len(total_video) < video_count:
                # [num_envs, height, width, channel]
                # last frame will be the first frame after automatic reset
                imgs = env.get_images() if t != max_timesteps - 1 else np.zeros_like(imgs)
                imgs = list(imgs)
                # add 'START' and 'END' text onto start frame and end frame of the images
                if t == 0:
                    imgs = addSTARTtoImages(imgs)
                else:
                    imgs = addSUCCESStoImages(
                        imgs, [bool(each['is_success']) for each in info])
                # per_video: [ts, num_envs, height, width, channel]
                per_video.append(imgs)
            # re-assign the observation =====================================================================================
            observation = observation_new
        total_success_rate += [each['is_success'] for each in info]
        total_distance += list(distance_between_rotations(
            final_observation['achieved_goal'][..., 3:], final_observation['desired_goal'][..., 3:]))
        total_reward += list(per_reward)
        if len(total_video) < video_count:
            # total_video: [num_video, ts, num_envs, height, width, channel]
            total_video.append(per_video)
        # store last timestep obs and goal
        mb_obs[episodes_collected:(
            episodes_collected + num_envs), max_timesteps, :] = np.array(observation_new[obs_key]).copy()
        mb_ag[episodes_collected:(
            episodes_collected + num_envs), max_timesteps, :] = np.array(observation_new['achieved_goal']).copy()
        episodes_collected += num_envs

    # redistribute metrics to each env =============================================
    total_success_rate = np.array(total_success_rate)
    total_distance = np.array(total_distance)
    total_reward = np.array(total_reward)
    assert total_success_rate.shape == total_distance.shape and total_distance.shape == total_reward.shape and total_reward.shape[
        0] == num_episodes
    total_success_rate_per_env, total_distance_per_env, total_reward_per_env, total_video_per_env = defaultdict(
        list), defaultdict(list), defaultdict(list), defaultdict(list)
    assert num_episodes == (num_episodes / num_envs) * len(vec_env_names)
    # iterate through all passes of vec envs
    for i in range(num_episodes // num_envs):
        # iterate through each pass of vec envs
        for j, env_name in enumerate(vec_env_names):
            # record metrics for each rollout
            total_success_rate_per_env[env_name].append(
                total_success_rate[i * num_envs + j])
            total_distance_per_env[env_name].append(
                total_distance[i * num_envs + j])
            total_reward_per_env[env_name].append(
                total_reward[i * num_envs + j])
    # average metrics across all their own rollouts
    total_success_rate_per_env = {env_name: np.mean(
        value) for env_name, value in total_success_rate_per_env.items()}
    total_distance_per_env = {env_name: np.mean(
        value) for env_name, value in total_distance_per_env.items()}
    total_reward_per_env = {env_name: np.mean(
        value) for env_name, value in total_reward_per_env.items()}

    # redistribute video to each env and convert to wandb format
    if video_count > 0:
        # [video_count, ts, num_vec_envs, height, width, channel] -> [num_vec_envs, video_count, ts, channel, height, width]
        total_video = np.transpose(np.array(total_video), (2, 0, 1, 5, 3, 4))
        assert total_video.shape[:4] == (
            num_envs, video_count, max_timesteps, 3)
        # redistribute
        for i, env_name in enumerate(vec_env_names):
            total_video_per_env[env_name].append(total_video[i])
        # flatten first two dims as they both contain separate videos
        total_video_per_env = {env_name: np.array(
            value) for env_name, value in total_video_per_env.items()}  # convert to numpy
        total_video_per_env = {env_name: np.array(
            value).reshape(-1, *value.shape[2:]) for env_name, value in total_video_per_env.items()}

    # calculate avg metrics
    total_success_rate_per_env['average'] = np.mean(total_success_rate)
    total_distance_per_env['average'] = np.mean(total_distance)
    total_reward_per_env['average'] = np.mean(total_reward)

    return mb_obs, mb_ag, mb_g, mb_actions, total_success_rate_per_env, total_reward_per_env, total_distance_per_env, total_video_per_env
