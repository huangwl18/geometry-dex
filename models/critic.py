import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.actor import MLPBase, PointnetMLP


class VanillaCritic(nn.Module):
    """a vanilla actor module that outputs a node's action given only its observation (no message between nodes)"""

    def __init__(self, state_dim, action_dim, max_action, args):
        super(VanillaCritic, self).__init__()
        self.max_action = max_action
        self.base = MLPBase(state_dim + action_dim, 1, args)

    def forward(self, x, u):
        x = torch.cat([x, u / self.max_action], dim=1)
        return self.base(x)


class PointnetMLP_critic(PointnetMLP):
    """a vanilla critic module + pointnet for processing point clouds"""

    def __init__(self, state_dim, action_dim, max_action, args):
        super().__init__(state_dim, 1, max_action, args)
        self.action_fc = nn.Linear(action_dim, args.pointnet_output_dim)
        self.value_net = MLPBase(
            3 * args.pointnet_output_dim, action_dim, args)

    def forward(self, states, actions):
        actions /= self.max_action
        state_context = self.features_net(states)
        action_feat = F.relu(self.action_fc(actions))
        action_feat = F.normalize(action_feat, dim=-1)
        context = torch.cat([state_context, action_feat], dim=-1)
        value = self.value_net(context)
        return value
