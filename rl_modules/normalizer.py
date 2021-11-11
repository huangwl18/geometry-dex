import threading
import numpy as np


class normalizer:
    def __init__(self, size, env_name=None, eps=1e-2, default_clip_range=np.inf, args=None):
        if env_name:
            self.env_name = env_name
        self.full_size = size
        self.num_points = args.num_points
        self.size = size
        self.args = args
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float64)
        self.local_sumsq = np.zeros(self.size, np.float64)
        self.local_count = np.zeros(1, np.float64)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float64)
        self.total_sumsq = np.zeros(self.size, np.float64)
        self.total_count = np.ones(1, np.float64)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

        # thread locker
        self.lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle lock
        del state["lock"]
        return state

    def __setstate__(self, state):
        del state['args']  # use current args instead of unpickled copy
        self.__dict__.update(state)
        # Add lock back since it doesn't exist in the pickle
        self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        if hasattr(self, 'args') and self.size != 7 and self.num_points > 0:
            v = v.reshape(-1, v.shape[-1])
            v = self.args.flat2dict(v)['minimal_obs']
        else:
            assert v.shape[-1] == self.size
            v = v.reshape(-1, v.shape[-1])

        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            try:
                squared = np.square(v)
            except FloatingPointError as e:
                print('### ENCOUNTER THE FOLLOWING FloatingPointError: ###')
                print(str(e))
                print('### re-attempting with rounded value ###')
                rounded = np.around(v, decimals=4)
                squared = np.square(rounded)
            self.local_sumsq += (squared).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        # update the total stuff
        self.total_sum += self.local_sum
        self.total_sumsq += self.local_sumsq
        self.total_count += self.local_count
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq /
                           self.total_count) - np.square(self.total_sum / self.total_count)))
        # cast back to single precision for consistency
        self.mean = self.mean.astype(np.float32)
        self.std = self.std.astype(np.float32)

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        if hasattr(self, 'num_points') and self.num_points > 0 and self.size != 7:
            v, points = v[..., :self.size], v[..., self.size:]
            # normalize v
            norm_v = np.clip((v - self.mean) / (self.std), -
                             clip_range, clip_range)
            return np.concatenate([norm_v, points], axis=-1)
        else:
            return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
