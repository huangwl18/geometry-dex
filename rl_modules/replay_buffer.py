import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class replay_buffer:
    def __init__(self, max_timesteps, obs_size, goal_size, action_size, buffer_size, env_name, sample_func):
        self.env_name = env_name
        self.T = max_timesteps
        self.size = buffer_size // self.T
        self.sample_func = sample_func
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, obs_size], dtype=np.float32),
                        'ag': np.empty([self.size, self.T + 1, goal_size], dtype=np.float32),
                        'g': np.empty([self.size, self.T, goal_size], dtype=np.float32),
                        'actions': np.empty([self.size, self.T, action_size], dtype=np.float32),
                        }
        # thread lock
        self.lock = threading.Lock()
        # for backward compatibility
        self.bc_agent = False
        self.return_next_action = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle lock
        del state["lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add lock back since it doesn't exist in the pickle
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # ignore the empty slots
    def get_valid_buffer(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = self.get_valid_buffer()
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(
            temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
