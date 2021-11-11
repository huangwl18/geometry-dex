import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet import *


class PointnetMLP(nn.Module):
    """a vanilla actor module + pointnet for processing point clouds"""

    def __init__(self, state_dim, action_dim, max_action, args):
        super(PointnetMLP, self).__init__()
        self.args = args
        self.max_action = max_action
        self.features_net = StatePointCloud(state_dim, args)
        self.action_net = MLPBase(
            2 * args.pointnet_output_dim, action_dim, args)

    def forward(self, x):
        context = self.features_net(x)
        action = self.action_net(context)
        action = self.max_action * torch.tanh(action)
        return action


class StatePointCloud(nn.Module):
    """return features of provided states and point clouds of current object and goal"""

    def __init__(self, state_dim, args):
        super(StatePointCloud, self).__init__()
        self.args = args
        # two-stream pointnet
        self.pointnet = PointNetfeatTwoStream(
            output_dim=args.pointnet_output_dim)  # expects [batch, 3, num_points]
        # projection layers
        self.points_fc = nn.Linear(
            args.pointnet_output_dim, args.pointnet_output_dim)
        self.states_fc = nn.Linear(state_dim, args.pointnet_output_dim)

    def forward(self, x):
        # slice inputs into different parts
        obs_dict = self.args.flat2dict(x)
        states, goal = obs_dict['minimal_obs'], obs_dict['desired_goal']
        # get sampled points ========================================================
        obj_points, target_points = obs_dict['object_points'], obs_dict['target_points']
        # reshape points
        assert len(obj_points.shape) == 2 and len(target_points.shape) == 2
        obj_points = obj_points.reshape(
            [x.shape[0], self.args.num_points, 3])
        target_points = target_points.reshape(
            [x.shape[0], self.args.num_points, 3])
        # get sampled normals ========================================================
        obj_normals, target_normals = obs_dict['object_normals'], obs_dict['target_normals']
        # reshape points
        assert len(obj_normals.shape) == 2 and len(
            target_normals.shape) == 2
        obj_normals = obj_normals.reshape(
            [x.shape[0], self.args.num_points, 3])
        target_normals = target_normals.reshape(
            [x.shape[0], self.args.num_points, 3])
        # get pointnet features ========================================================
        obj_points = torch.cat([obj_points, obj_normals], dim=-1)
        target_points = torch.cat(
            [target_points, target_normals], dim=-1)
        # need to do transpose in order to use the conv implementation of pointnet
        obj_points = obj_points.transpose(2, 1)
        target_points = target_points.transpose(2, 1)
        points_feat = self.pointnet(obj_points, target_points)
        # get global points feature from pointnet
        points_feat = F.relu(self.points_fc(points_feat))
        points_feat = F.normalize(points_feat, dim=-1)
        # get states features ========================================================
        # get states feature
        states_feat = F.relu(self.states_fc(torch.cat([states, goal], dim=-1)))
        states_feat = F.normalize(states_feat, dim=-1)

        return torch.cat([states_feat, points_feat], dim=-1)


class MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, args=None):
        super(MLPBase, self).__init__()
        # convert args to dict
        if not isinstance(args, dict):
            self.args = vars(args)
        elif args is None:
            self.args = dict()
        else:
            self.args = args
        # set default values if not provided
        if 'num_layers' not in self.args:
            self.args['num_layers'] = 4
        if 'width' not in self.args:
            self.args['width'] = 256
        assert self.args['num_layers'] >= 2

        self.layers = nn.ModuleList()
        for i in range(self.args['num_layers'] - 1):
            if i == 0:
                input_dim = num_inputs
            else:
                input_dim = self.args['width']
            self.layers.append(nn.Linear(input_dim, self.args['width']))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.args['width'], num_outputs))

    def forward(self, inputs):
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out


class VanillaActor(nn.Module):
    """a vanilla actor module that outputs a node's action given only its observation (no message between nodes)"""

    def __init__(self, state_dim, action_dim, max_action, args):
        super(VanillaActor, self).__init__()
        self.max_action = max_action
        self.base = MLPBase(state_dim, action_dim, args)
        self.args = args

    def forward(self, x):
        x = self.max_action * torch.tanh(self.base(x))
        return x
