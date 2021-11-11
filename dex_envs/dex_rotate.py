import os
import numpy as np
from gym import utils
from gym.envs.robotics.hand.manipulate import ManipulateEnv

OWN_PATH = os.path.dirname(os.path.abspath(__file__))


class a_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class chain_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class f_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class peach_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class chips_can_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class flute_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class pear_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class alarm_clock_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class c_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class foam_brick_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class phillips_screwdriver_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class a_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class cracker_box_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class fork_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class piggy_bank_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class a_marbles_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class c_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class f_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class pitcher_base_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class apple_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class cube_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class g_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class plum_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class a_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class cup_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class gelatin_box_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class potted_meat_can_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class ball_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class g_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class power_drill_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class banana_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class golf_ball_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class ps_controller_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class baseball_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class hammer_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class pudding_box_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class b_colored_wood_blocks_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class h_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class pyramid_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class b_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class headphones_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class racquetball_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class binoculars_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class h_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class rubber_duck_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class bleach_cleanser_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class i_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class rubiks_cube_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class b_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class i_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class scissors_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class i_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class small_marker_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class j_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class softball_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class d_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class j_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class sponge_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class dice_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class j_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class stanford_bunny_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class d_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class k_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class stapler_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class d_marbles_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class knife_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class strawberry_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class door_knob_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class k_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class sugar_box_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class d_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class large_clamp_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class tennis_ball_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class e_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class large_marker_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class timer_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class e_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class lemon_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class tomato_soup_can_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class elephant_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class light_bulb_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class toothbrush_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class l_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class toothpaste_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class master_chef_can_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class torus_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class bowl_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class medium_clamp_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class train_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class b_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class m_lego_duplo_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class tuna_fish_can_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class camera_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class mouse_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class utah_teapot_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class can_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class mug_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class water_bottle_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class e_toy_airplane_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class mustard_bottle_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class wine_glass_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class extra_large_clamp_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class nine_hole_peg_test_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class wood_block_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class f_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class orange_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class wristwatch_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class c_cups_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class flashlight_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class padlock_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class cell_phone_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class flat_screwdriver_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)


class pan_env(ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='ignore', target_rotation='z', reward_type='sparse', **kwargs):
        xml_path = os.path.join(OWN_PATH, 'assets', 'hand', 'manipulate_{}.xml'.format(self.__class__.__name__[:-4]))
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type, **kwargs)