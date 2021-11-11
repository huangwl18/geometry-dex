import copy
import numpy as np
from gym import Wrapper
from numpy.random import RandomState


class PointCloudWrapper(Wrapper):
    def __init__(self, env, args):
        super(PointCloudWrapper, self).__init__(env)
        self.env_name = env.spec.id[:-3]
        self._max_episode_steps = self.env._max_episode_steps
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.rand = RandomState(args.seed)
        self.args = args
        # rename the original obs to minimal_obs
        self.observation_space.spaces['minimal_obs'] = self.observation_space.spaces.pop(
            'observation')
        if args.point_cloud:
            self.observation_space.spaces['pc_obs'] = copy.deepcopy(
                self.observation_space.spaces['minimal_obs'])
            pc_dim = self.args.num_points * 6
            # shadowhands env also has target body, which also has point cloud as obs
            pc_dim *= 2
            self.observation_space.spaces['pc_obs'].shape = (
                self.observation_space.spaces['minimal_obs'].shape[0] + pc_dim,)

    def _normalize_points(self, point_set):
        """zero-center and scale to unit sphere"""
        point_set = point_set - \
            np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        return point_set

    def observation(self, observation):
        assert isinstance(observation, dict)
        observation['minimal_obs'] = observation.pop('observation')
        if self.args.point_cloud:
            assert len(self.env.sim.model.mesh_vertadr) == 13, '{} meshes found, expecting 13 (env: {})'.format(
                len(self.env.sim.model.mesh_vertadr), self.env_name)
            vert_start_adr = self.env.sim.model.mesh_vertadr[-1]
            object_vert = self.env.sim.model.mesh_vert[vert_start_adr:]
            # select some number of object vertices
            selected = self.rand.randint(
                low=0, high=object_vert.shape[0], size=self.args.num_points)
            sampled_points = object_vert[selected].copy()
            object_normals = self.env.sim.model.mesh_normal[vert_start_adr:]
            sampled_normals = object_normals[selected].copy()
            # transform to global coordinates
            object_xmat = self.env.sim.data.get_geom_xmat('object')
            object_points = np.matmul(sampled_points, object_xmat.T)
            object_points = self._normalize_points(object_points)
            object_normals = np.matmul(sampled_normals, object_xmat.T)
            target_xmat = self.env.sim.data.get_geom_xmat('target')
            target_points = np.matmul(sampled_points, target_xmat.T)
            target_points = self._normalize_points(target_points)
            target_normals = np.matmul(sampled_normals, target_xmat.T)
            # concat all obs
            observation['pc_obs'] = np.concatenate([observation['minimal_obs'],
                                                    object_points.flatten(),
                                                    object_normals.flatten(),
                                                    target_points.flatten(),
                                                    target_normals.flatten()])
        return observation

    def flat2dict(self, obs):
        """convert flat obs to dict"""
        state_dim = 61
        goal_dim = 7
        if obs.shape[-1] == state_dim:  # without goal
            return {'minimal_obs': obs}
        elif obs.shape[-1] == state_dim + goal_dim:  # with goal
            return {'minimal_obs': obs[..., :-goal_dim], 'desired_goal': obs[..., -goal_dim:]}
        # with normals, without goal
        elif obs.shape[-1] == state_dim + self.args.num_points * 12:
            return {'minimal_obs': obs[..., :state_dim],
                    'object_points': obs[..., state_dim:state_dim + self.args.num_points * 3],
                    'object_normals': obs[..., state_dim + self.args.num_points * 3:state_dim + self.args.num_points * 6],
                    'target_points': obs[..., state_dim + self.args.num_points * 6:state_dim + self.args.num_points * 9],
                    'target_normals': obs[..., state_dim + self.args.num_points * 9:]}
        # with normals, with goal
        elif obs.shape[-1] == state_dim + self.args.num_points * 12 + goal_dim:
            return {'minimal_obs': obs[..., :state_dim],
                    'object_points': obs[..., state_dim:state_dim + self.args.num_points * 3],
                    'object_normals': obs[..., state_dim + self.args.num_points * 3:state_dim + self.args.num_points * 6],
                    'target_points': obs[..., state_dim + self.args.num_points * 6:state_dim + self.args.num_points * 9],
                    'target_normals': obs[..., state_dim + self.args.num_points * 9:state_dim + self.args.num_points * 12],
                    'desired_goal': obs[..., -goal_dim:]}
        else:
            print(obs.shape)
            raise NotImplementedError

    def reset(self):
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info
