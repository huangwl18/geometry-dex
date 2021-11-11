import torch
import os
import numpy as np
import time
from models.actor import VanillaActor, PointnetMLP
from models.critic import VanillaCritic, PointnetMLP_critic
from rl_modules.replay_buffer import replay_buffer
from rl_modules.base_agent import base_agent
from rl_modules.normalizer import normalizer
from rl_modules.her import her_sampler
from rl_modules.utils import *

"""
ddpg with HER
"""


class ddpg_agent(base_agent):
    def __init__(self, args, dummy_env, env_params, policy_params):
        super().__init__(args, env_params)
        self.agent_type = 'ddpg_her'
        # create new run or resume from previous
        self._fresh_or_resume()
        self.args.flat2dict = dummy_env.flat2dict
        # sampling function (use her or not)
        # assuming all envs have the same compute reward method
        her_module = her_sampler(
            self.args.replay_strategy, self.args.replay_k, dummy_env.compute_reward)
        self.sample_func = her_module.sample_her_transitions
        self.buffers = None
        # create new modules or resumed from checkpoint
        if self.resume_training and self.args.eval and self.args.load_cycle is not None:  # load actor only for eval
            self.actor_network = self.ckpt_dict['actor_network']
            self.o_norm = self.ckpt_dict['o_norm']
            self.g_norm = self.ckpt_dict['g_norm']
            self.actor_network.args = self.args
            self.o_norm.args = self.args
            self.g_norm.args = self.args
        elif self.resume_training:  # load everything for resuming training
            self.actor_network = self.ckpt_dict['actor_network']
            self.actor_target_network = self.ckpt_dict['actor_target_network']
            self.critic_network = self.ckpt_dict['critic_network']
            self.critic_target_network = self.ckpt_dict['critic_target_network']
            self.actor_optim = self.ckpt_dict['actor_optim']
            self.critic_optim = self.ckpt_dict['critic_optim']
            self.o_norm = self.ckpt_dict['o_norm']
            self.g_norm = self.ckpt_dict['g_norm']
            self.actor_network.args = self.args
            self.actor_target_network.args = self.args
            self.critic_network.args = self.args
            self.critic_target_network.args = self.args
            self.o_norm.args = self.args
            self.g_norm.args = self.args
            if 'buffers' in self.ckpt_dict:
                self.buffers = self.ckpt_dict['buffers']
            else:
                assert self.args.eval
        else:
            self.actor_func = PointnetMLP if args.point_cloud else VanillaActor
            self.critic_func = PointnetMLP_critic if args.point_cloud else VanillaCritic
            self.actor_network = self.actor_func(**policy_params)
            self.critic_network = self.critic_func(**policy_params)
            self.actor_target_network = self.actor_func(**policy_params)
            self.actor_target_network.load_state_dict(
                self.actor_network.state_dict())
            self.critic_target_network = self.critic_func(**policy_params)
            self.critic_target_network.load_state_dict(
                self.critic_network.state_dict())
            if self.args.point_cloud:
                self._load_pointnet()
            # create the optimizer
            self.actor_optim = torch.optim.Adam(
                self.actor_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(
                self.critic_network.parameters(), lr=self.args.lr_critic)
            # use merged normalizer for all envs
            self.o_norm = normalizer(
                size=env_params['obs_to_normalize'], default_clip_range=self.args.clip_range, args=args)
            self.g_norm = normalizer(
                size=env_params['goal'], default_clip_range=self.args.clip_range, args=args)
        # create replay buffer
        if not self.args.eval and self.buffers is None:
            self.buffers = dict()
            for env_name in self.args.train_names:
                self.buffers[env_name] = replay_buffer(self.env_params['max_timesteps'], self.env_params['obs'], self.env_params['goal'],
                                                       self.env_params['action'], self.args.buffer_size, env_name=env_name,
                                                       sample_func=self.sample_func)
        self.device = torch.device("cuda" if not self.args.no_cuda else "cpu")
        # if use gpu
        if self.args.eval:
            if not self.args.no_cuda:
                self.actor_network.cuda()
            self.actor_network.eval()
        else:
            if not self.args.no_cuda:
                self.actor_network.cuda()
                self.critic_network.cuda()
                self.actor_target_network.cuda()
                self.critic_target_network.cuda()
            self.actor_network.eval()
            self.actor_target_network.eval()
            self.critic_network.eval()
            self.critic_target_network.eval()

    def learn(self, log_callback=None):
        """
        train the network

        """
        # start to collect samples
        assert self.args.num_rollouts % self.args.num_parallel_envs == 0
        if self.resume_training:
            self.total_steps = self.ckpt_dict['total_steps']
            start_cycle = self.ckpt_dict['current_cycle'] + 1
        else:
            self.total_steps = 0
            start_cycle = 0
        for cycle in range(start_cycle, self.args.n_cycles):
            self.current_cycle = cycle
            cycle_start = time.time()
            logged_dict = dict()
            sampling_start = time.time()
            # collect data ==================================================================
            train_env_reward, train_env_sr, train_env_distance = [], [], []
            # print('\n### Start collecting samples ###\n')
            mb_obs, mb_ag, mb_g, mb_actions, total_success_rate_per_env, total_reward_per_env, total_distance_per_env, _ = collect_experiences(self.train_envs, self.args.num_rollouts,
                                                                                                                                               self.env_params[
                                                                                                                                                   'max_timesteps'], self.actor_network,
                                                                                                                                               self.o_norm, self.g_norm, self.env_params,
                                                                                                                                               video_count=0, cuda=not self.args.no_cuda,
                                                                                                                                               action_proc_func=self._select_actions,
                                                                                                                                               vec_env_names=self.train_vec_env_names, point_cloud=self.args.point_cloud)
            # store the episodes
            assert self.args.num_rollouts == len(self.train_vec_env_names)
            assert mb_obs.shape[0] == self.args.num_rollouts
            for i, env_name in enumerate(self.train_vec_env_names):
                self.buffers[env_name].store_episode(
                    [mb_obs[i, :, :], mb_ag[i, :, :], mb_g[i, :, :], mb_actions[i, :, :]])
            # update normalizer
            self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
            # print to console =================================================
            print('== [EXP{:04d}:C{:03d}] success %: {:.3f}, reward: {:.2f}, dist: {:.3f}, env time: {:.2f} s across {} episodes'.format(self.args.expID, cycle,
                                                                                                                                         total_success_rate_per_env['average'], total_reward_per_env['average'], total_distance_per_env['average'], time.time() - sampling_start, mb_actions.shape[0]))
            # accumulate metrics
            train_env_reward.append(total_reward_per_env['average'])
            train_env_distance.append(total_distance_per_env['average'])
            train_env_sr.append(total_success_rate_per_env['average'])
            self.total_steps += self.args.num_rollouts * \
                self.env_params['max_timesteps']
            logged_dict['avg_train_reward_train'] = np.mean(train_env_reward)
            logged_dict['avg_train_dist_train'] = np.mean(train_env_distance)
            logged_dict['avg_train_sr_train'] = np.mean(train_env_sr)
            logged_dict['total_env_steps'] = self.total_steps
            # start training ==================================================================
            training_start = time.time()
            update_infos = []
            num_multi_batch = self.args.n_batches // len(self.args.train_names)
            for b in range(num_multi_batch):  # each iter trains all envs once
                # train the network
                update_info = self._update_network()
                update_infos.append(update_info)
            # average and log training metrics
            logged_dict.update(get_avg_values_across_batches(update_infos))
            # update targets
            self._soft_update_target_network(
                self.actor_target_network, self.actor_network)
            self._soft_update_target_network(
                self.critic_target_network, self.critic_network)
            logged_dict['fps'] = self.args.num_rollouts * \
                self.env_params['max_timesteps'] // (time.time() - cycle_start)
            print('{:03d}:{} -- actor/l: {:.3f}, critic/l: {:.3f}, steps: {:.3e}, Cycle FPS: {}, train time: {:.2f} s -----\n'.format(self.args.expID,
                                                                                                                                      cycle, logged_dict['actor_loss'], logged_dict['critic_loss'], self.total_steps, logged_dict['fps'], time.time() - training_start))
            # start to do the evaluation
            if self.args.eval_freq > 0 and (cycle % self.args.eval_freq == 0 or cycle == 0 or cycle == self.args.n_cycles - 1):
                logged_dict = self._eval_network(logged_dict)
            if log_callback is not None:
                log_callback(logged_dict)

    def _select_actions(self, pi):
        """this function will choose action for the agent and do the exploration"""
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * \
            self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(
            action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'],
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps,
                                     1)[0] * (random_actions - action)
        return action

    def _update_normalizer(self, episode_batch):
        """update the normalizer"""
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.sample_func(buffer_temp, num_transitions)
        # pre process the obs and g
        transitions['obs'], transitions['g'] = preproc_og(
            transitions['obs'], transitions['g'], clip_obs=self.args.clip_obs)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _soft_update_target_network(self, target, source):
        """update targets"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _get_train_data(self, env_name):
        # sample the episodes
        transitions = self.buffers[env_name].sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = preproc_og(
            o, g, clip_obs=self.args.clip_obs)
        transitions['obs_next'], transitions['g_next'] = preproc_og(
            o_next, g, clip_obs=self.args.clip_obs)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=-1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate(
            [obs_next_norm, g_next_norm], axis=-1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(
            inputs_norm, dtype=torch.float32, device=self.device)
        inputs_next_norm_tensor = torch.tensor(
            inputs_next_norm, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(
            transitions['actions'], dtype=torch.float32, device=self.device)
        r_tensor = torch.tensor(
            transitions['r'], dtype=torch.float32, device=self.device)
        return inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor

    # update the network
    def _update_network(self):
        # set models to train mode
        self.actor_network.train()
        self.actor_target_network.train()
        self.critic_network.train()
        self.critic_target_network.train()
        # train each object
        critic_losses, actor_losses = [], []
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        for i, env_name in enumerate(self.args.train_names):
            inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor = self._get_train_data(
                env_name)
            # calculate the target Q value function
            with torch.no_grad():
                # concatenate the stuffs
                actions_next = self.actor_target_network(
                    inputs_next_norm_tensor)
                q_next_value = self.critic_target_network(
                    inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                # clip the q value
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            # the q loss
            real_q_value = self.critic_network(
                inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()
            critic_losses.append(critic_loss)
            # the actor loss
            actions_real = self.actor_network(inputs_norm_tensor)
            actor_loss = - \
                self.critic_network(inputs_norm_tensor, actions_real).mean()
            action_l2_norm = (
                actions_real / self.env_params['action_max']).pow(2).mean()
            actor_loss += self.args.action_l2 * action_l2_norm
            actor_losses.append(actor_loss)
        actor_loss = sum(actor_losses)
        critic_loss = sum(critic_losses)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # set models to eval mode
        self.actor_network.eval()
        self.actor_target_network.eval()
        self.critic_network.eval()
        self.critic_target_network.eval()

        update_info = dict(actor_loss=actor_loss, critic_loss=critic_loss)
        return update_info

    def _save_ckpt(self, exp_path):
        # save general checkpoint for training resume
        general_checkpoint = dict(actor_network=self.actor_network,
                                  actor_target_network=self.actor_target_network,
                                  critic_network=self.critic_network,
                                  critic_target_network=self.critic_target_network,
                                  actor_optim=self.actor_optim,
                                  critic_optim=self.critic_optim,
                                  o_norm=self.o_norm,
                                  g_norm=self.g_norm,
                                  current_cycle=self.current_cycle,
                                  total_steps=self.total_steps,
                                  args=self.args)
        if not self.args.no_save_buffer:
            general_checkpoint['buffers'] = self.buffers
        general_save_path = os.path.join(
            exp_path, '{}_general_{:04d}.tar'.format(self.agent_type, self.args.expID))
        torch.save(general_checkpoint, general_save_path)
        print('** general checkpoint saved to {}'.format(general_save_path))

        eval_checkpoint = dict(actor_network=self.actor_network,
                               o_norm=self.o_norm,
                               g_norm=self.g_norm)
        eval_save_path = os.path.join(exp_path, '{}_eval_{:04d}_J{}.tar'.format(
            self.agent_type, self.args.expID, self.current_cycle))
        torch.save(eval_checkpoint, eval_save_path)
        print('** eval checkpoint saved to {}'.format(eval_save_path))

    def _load_pointnet(self):
        pointnet_load_path = './dex_logs/pointnet/pointnet_{:04d}'.format(
            self.args.pointnet_load_path)
        fname = 'feat_model.pth'
        pretrain_weights = torch.load(os.path.join(pointnet_load_path, fname))
        for nn_name in ['actor_network', 'critic_network', 'actor_target_network', 'critic_target_network']:
            nn = getattr(self, nn_name)
            nn.features_net.pointnet.load_state_dict(pretrain_weights)
            if not self.args.finetune_pointnet:
                for name, p in nn.features_net.pointnet.named_parameters():
                    p.requires_grad = False
        print('*** successfully loaded pointnet feature model ***')
