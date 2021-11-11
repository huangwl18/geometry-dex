
import os
import numpy as np
from rl_modules.utils import *
import wandb
import time
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_chunk_vec_env import SubprocChunkVecEnv

"""
base agent
"""


class base_agent:
    def __init__(self, args, env_params):
        self.args = args
        self.num_train_envs = len(self.args.train_names)
        self.num_test_envs = len(self.args.test_names)
        self.env_params = env_params
        # create training envs for all train objects ==============================================================================
        if self.num_train_envs > 0:
            assert self.args.num_parallel_envs % self.num_train_envs == 0
            self.num_repeated = self.args.num_parallel_envs // self.num_train_envs
            start = time.time()
            print(
                '\n** creating {} training vec env.. '.format(self.args.num_parallel_envs), end='')
            # construct make env functions
            make_fns = []
            self.train_vec_env_names = []
            count = 0
            for env_name in self.args.train_names:
                for _ in range(self.num_repeated):
                    make_fns.append(makeEnv(env_name, count, self.args))
                    self.train_vec_env_names.append(env_name)
                    count += 1
            assert len(make_fns) == self.args.num_parallel_envs
            # env parallelization methods (recommend chunk envs with chunk_size = ~10)
            if self.args.serial:
                self.train_envs = DummyVecEnv(make_fns)
            elif self.args.chunk_size != 0:
                self.train_envs = SubprocChunkVecEnv(
                    make_fns, self.args.chunk_size)
            else:
                self.train_envs = SubprocVecEnv(make_fns)
            print(
                'created (time taken: {:.2f} s)! **'.format(time.time() - start))
        # create eval envs for all train + test objects ==============================================================================
        start = time.time()
        print('\n** creating {} eval vec env.. '.format(self.args.n_test_rollouts), end='')
        # construct make env functions
        eval_num_repeated = self.args.n_test_rollouts // (
            self.num_train_envs + self.num_test_envs)
        make_fns = []
        self.eval_vec_env_names = []
        count = 0
        for env_name in (self.args.train_names + self.args.test_names):
            for _ in range(eval_num_repeated):
                make_fns.append(makeEnv(env_name, count, self.args))
                self.eval_vec_env_names.append(env_name)
                count += 1
        # env parallelization methods (recommend chunk envs with chunk_size = ~10)
        if self.args.serial:
            self.eval_envs = DummyVecEnv(make_fns)
        elif self.args.chunk_size != 0:
            # limit the maximum parallel envs assigned to each cpu to 15 for eval envs
            num_cpu = len(os.sched_getaffinity(0))
            num_eval_envs = len(self.eval_vec_env_names)
            self.eval_envs = SubprocChunkVecEnv(
                make_fns, min(15, max(3, num_eval_envs // num_cpu)))
        else:
            self.eval_envs = SubprocVecEnv(make_fns)
        print(
            'created (time taken: {:.2f} s)! **\n'.format(time.time() - start))

    def eval(self, log_callback=None):
        """eval trained agent on test envs"""
        self.actor_network.eval()
        logged_dict = self._eval_network()
        if log_callback is not None:
            log_callback(logged_dict)

    def _eval_network(self, logged_dict=dict()):
        start = time.time()
        print('*' * 40)
        # eval on train + test envs ==================================================
        print('** evaluating over {} episodes each env over {} train + test envs'.format(
            self.args.n_test_rollouts, self.num_train_envs + self.num_test_envs))
        assert self.args.video_count <= 1, 'currently only support up to 1 video per env'
        video_num = 1 if self.args.video_count > 0 else 0
        _, _, _, _, total_success_rate_per_env, total_reward_per_env, total_distance_per_env, total_video_per_env = collect_experiences(self.eval_envs, self.args.n_test_rollouts, self.env_params['max_timesteps'],
                                                                                                                                        self.actor_network, self.o_norm, self.g_norm, self.env_params,
                                                                                                                                        cuda=not self.args.no_cuda, video_count=video_num, point_cloud=self.args.point_cloud, vec_env_names=self.eval_vec_env_names)
        # logging metrics ==================================================
        if self.num_train_envs:
            logged_dict['avg_eval_reward_train'] = np.mean(
                [total_reward_per_env[env_name] for env_name in self.args.train_names])
            logged_dict['avg_eval_dist_train'] = np.mean(
                [total_distance_per_env[env_name] for env_name in self.args.train_names])
            logged_dict['avg_eval_sr_train'] = np.mean(
                [total_success_rate_per_env[env_name] for env_name in self.args.train_names])
            print('** train_envs -- eval success %: {:.4f} eval reward: {:.2f}'.format(
                logged_dict['avg_eval_sr_train'], logged_dict['avg_eval_reward_train']))
        if self.num_test_envs:
            logged_dict['avg_eval_reward_test'] = np.mean(
                [total_reward_per_env[env_name] for env_name in self.args.test_names])
            logged_dict['avg_eval_dist_test'] = np.mean(
                [total_distance_per_env[env_name] for env_name in self.args.test_names])
            logged_dict['avg_eval_sr_test'] = np.mean(
                [total_success_rate_per_env[env_name] for env_name in self.args.test_names])
            print('** test_envs -- eval success %: {:.4f} eval reward: {:.2f}'.format(
                logged_dict['avg_eval_sr_test'], logged_dict['avg_eval_reward_test']))
        # logging video ==================================================
        if video_num > 0:
            try:
                for i in range(video_num):
                    for env_name in self.args.train_names:
                        logged_dict['{}-eval-video-{}-train'.format(env_name, i)] = wandb.Video(
                            total_video_per_env[env_name][i], fps=10, format="mp4")
                    for env_name in self.args.test_names:
                        logged_dict['{}-eval-video-{}-test'.format(env_name, i)] = wandb.Video(
                            total_video_per_env[env_name][i], fps=10, format="mp4")
                print(
                    '** generated {} eval videos per env to wandb'.format(video_num))
            except Exception as e:
                print('** ------ got the following error when generating video -------')
                print(e)
                print('** ------ ignoring video generation this time ----------------')
        # save models ==================================================
        if not self.args.no_save:
            self._save_ckpt(self.exp_path)
        print('** used memory: {:.2f} MB'.format(get_memory_usage()))
        print('** time taken: {:.2f} s'.format(time.time() - start))
        print('*' * 40)
        return logged_dict

    def _fresh_or_resume(self):
        # create method directory
        assert self.agent_type is not None
        self.args.agent_type_dir = os.path.join(
            self.args.save_dir, self.agent_type)
        if not os.path.exists(self.args.agent_type_dir):
            os.makedirs(self.args.agent_type_dir)
        # check for existing log
        expID = self.args.load_path if self.args.eval else self.args.expID
        self.exp_path = os.path.join(
            self.args.agent_type_dir, '{}_{:04d}'.format(self.agent_type, expID))
        if self.args.load_cycle is not None:
            checkpoint_exist = os.path.exists(os.path.join(
                self.exp_path, '{}_eval_{:04d}_J{}.tar'.format(self.agent_type, expID, self.args.load_cycle)))
        else:
            checkpoint_exist = os.path.exists(os.path.join(
                self.exp_path, '{}_general_{:04d}.tar'.format(self.agent_type, expID)))
        if self.args.eval:
            assert checkpoint_exist, 'checkpoint not found for evaluation'
        if self.args.eval or (not self.args.fresh and checkpoint_exist):
            self.resume_training = True
            self.ckpt_dict = self._load_from_ckpt(self.exp_path)
            print('*' * 40)
            print('** start from loaded checkpoint')
            print('*' * 40)
        else:
            if not os.path.exists(self.exp_path):
                os.makedirs(self.exp_path)
            self.resume_training = False
            print('*' * 40)
            print('** starting a new run')
            print('*' * 40)
        return self.resume_training

    def _save_ckpt(self, exp_path):
        pass

    def _load_from_ckpt(self, exp_path):
        expID = self.args.load_path if self.args.eval else self.args.expID
        if self.args.load_cycle is not None:
            cycle_load_path = os.path.join(
                exp_path, '{}_eval_{:04d}_J{}.tar'.format(self.agent_type, expID, self.args.load_cycle))
            return torch.load(cycle_load_path)
        else:
            general_load_path = os.path.join(
                exp_path, '{}_general_{:04d}.tar'.format(self.agent_type, expID))
            return torch.load(general_load_path)

    def learn(self, log_callback=None):
        pass

    # update the network
    def _update_network(self, input_norm, action_label):
        pass
