from rl_modules.utils import *
import torch
import random
from rl_modules.ddpg_agent import ddpg_agent
from arguments_ddpg import get_args
import os
import numpy as np
import dex_envs
import wandb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
train the agent
"""


def init_callback(args, prefix):
    if not args.eval and not args.fresh:
        resume_mode = 'allow'
    else:
        resume_mode = None
    run_name = '{}_{:04d}'.format(prefix, args.expID)
    wandb.init(name=run_name, id=run_name, resume=resume_mode,
               save_code=True, anonymous="allow")
    wandb.config.update(args, allow_val_change=True)


def log_callback(log_dict):
    wandb.log(log_dict)


def get_env_params(env, args):
    obs = env.reset()
    # close the environment
    params = {'goal': obs['desired_goal'].shape[-1],
              'action': env.action_space.shape[-1],
              'action_max': env.action_space.high[-1],
              'max_timesteps': env._max_episode_steps,
              'obs_to_normalize': obs['minimal_obs'].shape[-1]
              }
    if args.point_cloud:
        params['obs'] = obs['pc_obs'].shape[-1]
    else:
        params['obs'] = params['obs_to_normalize']
    return params


def get_policy_params(env, args):
    obs = env.reset()
    params = dict(state_dim=obs['minimal_obs'].shape[-1] + obs['desired_goal'].shape[-1],
                  action_dim=env.action_space.shape[-1],
                  max_action=env.action_space.high[-1],
                  args=args)
    return params


def launch(init_callback=None, log_callback=None):
    args = get_args()
    # create dummy env for accessing spaces attr
    dummy_env = makeEnv((args.train_names + args.test_names)[0], 0, args)()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    # assume all envs high-level attributes are the same, use arbitrary one
    env_params = get_env_params(dummy_env, args)
    # assume all envs high-level attributes are the same, use arbitrary one
    policy_params = get_policy_params(dummy_env, args)
    # create the ddpg agent to interact with the environment
    trainer = ddpg_agent(args, dummy_env, env_params, policy_params)
    init_callback(args=args, prefix=trainer.agent_type)

    if args.eval:
        trainer.eval(log_callback=log_callback)
    else:
        trainer.learn(log_callback=log_callback)

    dummy_env.close()


if __name__ == "__main__":
    # env setting ========================================================================
    # do not enable wandb output
    os.environ["WANDB_SILENT"] = "true"
    launch(init_callback=init_callback, log_callback=log_callback)
