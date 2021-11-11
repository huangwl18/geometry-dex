import os
from gym.envs.registration import register

OWN_PATH = os.path.dirname(os.path.abspath(__file__))


# Arguments for different versions of dex-rotate environments, used to initialize ManipulateEnv from Gym
all_versions = {
    'v0': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'sparse', 'randomize_initial_position': True, 'randomize_initial_rotation': True},
    'v1': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'sparse', 'randomize_initial_position': True, 'randomize_initial_rotation': True},
    'v2': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'dense', 'randomize_initial_position': True, 'randomize_initial_rotation': True},
    'v3': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense', 'randomize_initial_position': True, 'randomize_initial_rotation': True},
    'v4': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'sparse', 'randomize_initial_position': False, 'randomize_initial_rotation': True},
    'v5': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'sparse', 'randomize_initial_position': False, 'randomize_initial_rotation': True},
    'v6': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'dense', 'randomize_initial_position': False, 'randomize_initial_rotation': True},
    'v7': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense', 'randomize_initial_position': False, 'randomize_initial_rotation': True},
    'v8': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'sparse', 'randomize_initial_position': True, 'randomize_initial_rotation': False},
    'v9': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'sparse', 'randomize_initial_position': True, 'randomize_initial_rotation': False},
    'v10': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'dense', 'randomize_initial_position': True, 'randomize_initial_rotation': False},
    'v11': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense', 'randomize_initial_position': True, 'randomize_initial_rotation': False},
    'v12': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'sparse', 'randomize_initial_position': False, 'randomize_initial_rotation': False},
    'v13': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'sparse', 'randomize_initial_position': False, 'randomize_initial_rotation': False},
    'v14': {'target_position': 'ignore', 'target_rotation': 'xyz', 'reward_type': 'dense', 'randomize_initial_position': False, 'randomize_initial_rotation': False},
    'v15': {'target_position': 'ignore', 'target_rotation': 'z', 'reward_type': 'dense', 'randomize_initial_position': False, 'randomize_initial_rotation': False},
}

version = 'v1'
version_args = all_versions[version]

for fname in os.listdir(os.path.join(OWN_PATH, 'assets/hand/')):
    if 'manipulate_' in fname and '.xml' in fname:
        obj_name = fname[11:-4]
    else:
        continue
    env_class_name = obj_name + '_env'
    try:
        env_id = '{}-rotate-{}'.format(obj_name, version)
        register(
            id=env_id,
            entry_point='dex_envs.dex_rotate:{}'.format(env_class_name),
            kwargs=version_args,
            max_episode_steps=100,
        )
    except Exception as e:
        print('** ------ got the following error during env registration -------')
        print('erorr encountered when registering {}'.format(obj_name))
