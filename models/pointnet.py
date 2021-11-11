from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, input_dim=3, output_dim=1024):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        return out3  # return non-aggregated per-point features


class PointNetfeatTwoStream(nn.Module):
    """return a global feature vector describing the two input point clouds (same with point-to-point correspondence but rotated with a unknown matrix)"""

    def __init__(self, output_dim=1024, **kwargs):
        super(PointNetfeatTwoStream, self).__init__()
        self.output_dim = output_dim
        # pointnet model for extracting local (per-point) features
        self.feat_net = PointNetfeat(input_dim=12, output_dim=output_dim)

    def aggregate(self, x):
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, self.output_dim)

    def forward(self, point_set1, point_set2):
        """shape of input: [num_samples, 3, num_points]"""
        local_features = self.feat_net(
            torch.cat([point_set1, point_set2], dim=1))
        global_features = self.aggregate(local_features)
        return global_features


class PointNetClsAndPose(nn.Module):
    def __init__(self, num_classes=2, output_dim=1024):
        super(PointNetClsAndPose, self).__init__()
        # determine whether to fuse per-point features
        self.feat_net = PointNetfeatTwoStream(output_dim=output_dim)
        # classification head
        self.cls_fc1 = nn.Linear(output_dim, 512)
        self.cls_fc2 = nn.Linear(512, 256)
        self.cls_fc3 = nn.Linear(256, num_classes)
        self.cls_bn1 = nn.BatchNorm1d(512)
        self.cls_bn2 = nn.BatchNorm1d(256)
        self.cls_dropout = nn.Dropout(p=0.3)
        # regression head
        self.rotation_fc1 = nn.Linear(output_dim, 512)
        self.rotation_fc2 = nn.Linear(512, 256)
        self.rotation_fc3 = nn.Linear(256, 9)
        self.rotation_bn1 = nn.BatchNorm1d(512)
        self.rotation_bn2 = nn.BatchNorm1d(256)
        self.rotation_dropout = nn.Dropout(p=0.3)

    def forward(self, point_set1, point_set2):
        """shape of input: [num_samples, 3, num_points]"""
        global_features = self.feat_net(point_set1, point_set2)
        # only predict class for first point cloud
        pred_class1 = F.relu(self.cls_bn1(self.cls_fc1(global_features)))
        pred_class1 = F.relu(self.cls_bn2(
            self.cls_dropout(self.cls_fc2(pred_class1))))
        pred_class1 = self.cls_fc3(pred_class1)
        pred_class1 = F.log_softmax(pred_class1, dim=1)
        # predict transformation matrix between first and second
        pred_rotation = F.relu(self.rotation_bn1(
            self.rotation_fc1(global_features)))
        pred_rotation = F.relu(self.rotation_bn2(
            self.rotation_dropout(self.rotation_fc2(pred_rotation))))
        pred_rotation = self.rotation_fc3(pred_rotation)

        return pred_class1, pred_rotation
