from __future__ import print_function
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from models.pointnet import PointNetClsAndPose
import torch.nn.functional as F
import gym
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from numpy.random import RandomState
import numpy as np
import wandb
from arguments_pointnet import get_args
import dex_envs


def get_accuracy(pred_class, target):
    pred_choice = pred_class.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    accuracy = correct.item() / float(args.batch_size)
    return accuracy


def arcsine_loss(pred_rotation, rotation):
    assert len(pred_rotation.shape) == 2 and pred_rotation.shape[1] == 9
    assert len(rotation.shape) == 2 and rotation.shape[1] == 9
    frobenius = torch.sqrt(torch.sum((pred_rotation - rotation) ** 2, dim=1))
    loss = 2 * \
        torch.arcsin(torch.minimum(torch.ones_like(
            frobenius), frobenius / (2 * np.sqrt(2))))
    return loss.mean(dim=0)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1)/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1)
    theta = torch.acos(cos)
    return theta


class TwoStreamDataset(Dataset):
    def __init__(self, env_names, num_points=2500, data_aug=True, std_data_aug=0.02, num_data=100000, seed=123):
        self.envs = [makeEnv(env_name, 0, args)() for env_name in env_names]
        self.num_classes = len(env_names)
        for env in self.envs:
            env.reset()
            assert len(env.sim.model.mesh_vertadr) == 13, '{} meshes found, expecting 13 (env: {})'.format(
                len(env.sim.model.mesh_vertadr), env)
        self.rand = RandomState(seed)
        # self.rand = np.random
        self.num_points = num_points
        self.data_aug = data_aug
        self.num_data = num_data
        self.std_data_aug = std_data_aug

    def _get_points(self, env):
        vert_start_adr = env.sim.model.mesh_vertadr[-1]
        object_vert = env.sim.model.mesh_vert[vert_start_adr:]
        # select some number of object vertices
        selected = self.rand.randint(
            low=0, high=object_vert.shape[0], size=self.num_points)
        sampled_points = object_vert[selected].copy()
        assert sampled_points.shape[0] == self.num_points and sampled_points.shape[1] == 3
        object_normals = env.sim.model.mesh_normal[vert_start_adr:]
        sampled_normals = object_normals[selected].copy()
        assert sampled_normals.shape[0] == self.num_points and sampled_normals.shape[1] == 3
        return sampled_points, sampled_normals

    def _normalize(self, point_set):
        """zero-center and scale to unit sphere"""
        point_set = point_set - \
            np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        return point_set

    def _augment(self, point_set):
        # random jitter
        point_set += self.rand.normal(0,
                                      self.std_data_aug, size=point_set.shape)
        return point_set

    def __getitem__(self, index):
        target = index % self.num_classes
        sampled_points, sampled_normals = self._get_points(
            self.envs[target])
        # apply random rotation to first point set
        first_rotation = R.random()
        point_set1 = np.matmul(sampled_points, first_rotation.as_dcm().T)
        # apply random rotation to second point set
        second_rotation = R.random()
        point_set2 = np.matmul(sampled_points, second_rotation.as_dcm().T)
        # apply same rotations to normals
        normal_set1 = np.matmul(sampled_normals, first_rotation.as_dcm().T)
        normal_set2 = np.matmul(
            sampled_normals, second_rotation.as_dcm().T)
        # obtain the rotation between two rotated point sets
        rotation_diff = np.matmul(
            second_rotation.as_dcm(), first_rotation.inv().as_dcm())
        rotation_diff = rotation_diff.astype(
            np.float32).flatten()  # reformat for training

        # zero-center and scale to unit sphere
        point_set1 = self._normalize(point_set1)
        point_set2 = self._normalize(point_set2)

        # data augmentation
        if self.data_aug:
            point_set1 = self._augment(point_set1)
            point_set2 = self._augment(point_set2)

        return_set1 = np.concatenate(
            [point_set1, normal_set1], axis=-1).astype(np.float32)
        return_set2 = np.concatenate(
            [point_set2, normal_set2], axis=-1).astype(np.float32)

        return return_set1, return_set2, target, rotation_diff

    def __len__(self):
        return int(self.num_data)


def get_dataloaders(args):

    train_dataset = TwoStreamDataset(args.train_names, num_points=args.num_points, data_aug=True,
                                     std_data_aug=args.std_data_aug, seed=args.seed)
    val_dataset = TwoStreamDataset(args.train_names, num_points=args.num_points,
                                   data_aug=False, seed=args.seed + 1)
    heldout_dataset = TwoStreamDataset(args.test_names, num_points=args.num_points,
                                       data_aug=False, seed=args.seed + 2)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True)
    heldout_dataloader = torch.utils.data.DataLoader(
        heldout_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True)

    return train_dataloader, val_dataloader, heldout_dataloader


def train_cls_pose(args):
    train_dataloader, val_dataloader, heldout_dataloader = get_dataloaders(
        args)
    model = PointNetClsAndPose(num_classes=len(
        args.train_names), output_dim=args.output_dim)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_batch = len(train_dataloader)
    device = torch.device("cuda")
    count = 0
    for epoch in range(args.n_epoch):
        scheduler.step()
        for i, data in enumerate(train_dataloader):
            logged_dict = dict()
            point_set1, point_set2, target, rotation = data
            point_set1 = point_set1.to(device, non_blocking=True)
            point_set2 = point_set2.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            rotation = rotation.to(device, non_blocking=True)
            # conv implementation needs transpose
            point_set1 = point_set1.transpose(2, 1)
            point_set2 = point_set2.transpose(2, 1)
            optimizer.zero_grad()
            pred_class, pred_rotation = model(point_set1, point_set2)
            cls_loss = F.nll_loss(pred_class, target)
            pose_loss = arcsine_loss(pred_rotation, rotation)
            loss = args.alpha * cls_loss + (1 - args.alpha) * pose_loss
            loss.backward()
            optimizer.step()
            # calculate accuracy
            accuracy = get_accuracy(pred_class, target)
            geodesic = compute_geodesic_distance_from_two_matrices(
                pred_rotation.reshape(-1, 3, 3), rotation.reshape(-1, 3, 3)).mean(axis=0)
            print('[{}: {}/{}] cls loss: {:.3f}, pose loss: {:.3f}, loss: {:.3f}, accuracy: {:.3f}, geodesic: {:.3f}'.format(
                epoch, i, num_batch, cls_loss.item(), pose_loss.item(), loss.item(), accuracy, geodesic))
            logged_dict['train_cls_loss'] = cls_loss
            logged_dict['train_pose_loss'] = pose_loss
            logged_dict['train_loss'] = loss
            logged_dict['train_accuracy'] = accuracy
            logged_dict['train_geodesic'] = geodesic
            # eval
            if count % 100 == 0:
                print('*' * 40)
                model = model.eval()
                with torch.no_grad():
                    val_data = next(iter(val_dataloader))
                    # eval over the training objects
                    point_set1, point_set2, target, rotation = val_data
                    point_set1 = point_set1.to(device, non_blocking=True)
                    point_set2 = point_set2.to(device, non_blocking=True)
                    # conv implementation needs transpose
                    point_set1 = point_set1.transpose(2, 1)
                    point_set2 = point_set2.transpose(2, 1)
                    target = target.to(device, non_blocking=True)
                    rotation = rotation.to(device, non_blocking=True)
                    pred_class, pred_rotation = model(point_set1, point_set2)
                    cls_loss = F.nll_loss(pred_class, target)
                    pose_loss = ((rotation - pred_rotation) ** 2).mean()
                    loss = args.alpha * cls_loss + (1 - args.alpha) * pose_loss
                    accuracy = get_accuracy(pred_class, target)
                    geodesic = compute_geodesic_distance_from_two_matrices(
                        pred_rotation.reshape(-1, 3, 3), rotation.reshape(-1, 3, 3)).mean(axis=0)
                    print('** VAL: [{}: {}/{}] cls loss: {:.3f}, pose loss: {:.3f}, loss: {:.3f}, accuracy: {:.3f}, geodesic: {:.3f}'.format(
                        epoch, i, num_batch, cls_loss.item(), pose_loss.item(), loss.item(), accuracy, geodesic))
                    logged_dict['val_cls_loss'] = cls_loss
                    logged_dict['val_pose_loss'] = pose_loss
                    logged_dict['val_loss'] = loss
                    logged_dict['val_accuracy'] = accuracy
                    logged_dict['val_geodesic'] = geodesic
                    # eval over the test objects
                    heldout_data = next(iter(heldout_dataloader))
                    point_set1, point_set2, target, rotation = heldout_data
                    point_set1 = point_set1.to(device, non_blocking=True)
                    point_set2 = point_set2.to(device, non_blocking=True)
                    # conv implementation needs transpose
                    point_set1 = point_set1.transpose(2, 1)
                    point_set2 = point_set2.transpose(2, 1)
                    target = target.to(device, non_blocking=True)
                    rotation = rotation.to(device, non_blocking=True)
                    pred_class, pred_rotation = model(point_set1, point_set2)
                    pose_loss = ((rotation - pred_rotation) ** 2).mean()
                    geodesic = compute_geodesic_distance_from_two_matrices(
                        pred_rotation.reshape(-1, 3, 3), rotation.reshape(-1, 3, 3)).mean(axis=0)
                    print('** HELD-OUT: [{}: {}/{}] pose loss: {:.3f}, geodesic: {:.3f}'.format(
                        epoch, i, num_batch, pose_loss.item(), geodesic))
                    logged_dict['heldout_pose_loss'] = pose_loss
                    logged_dict['heldout_geodesic'] = geodesic
                model = model.train()
                print('*' * 40)
            log_callback(logged_dict)
            count += 1
        if not args.no_save:
            torch.save(model.state_dict(), '%s/cls_pose_model.pth' %
                       (args.save_path))
            torch.save(model.feat_net.state_dict(),
                       '%s/feat_model.pth' % (args.save_path))


def create_save_dirs(args):
    agent_type = 'pointnet'
    # create method directory
    args.save_dir = os.path.join(args.save_dir, agent_type)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # check for existing log
    exp_path = os.path.join(
        args.save_dir, '{}_{:04d}'.format(agent_type, args.expID))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    args.save_path = exp_path
    print('*' * 40)
    print('** starting a new run')
    print('*' * 40)


def makeEnv(env_name, idx, args):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        e = gym.make('{}-rotate-v1'.format(env_name))
        e.seed(args.seed + idx)
        return e
    return helper


def init_wandb(args, run_name):
    wandb.init(name=run_name, id=run_name, resume=None,
               save_code=True, anonymous="allow")


def log_callback(logged_dict):
    wandb.log(logged_dict)


if __name__ == '__main__':
    # env setting ========================================================================
    # always raise numpy error
    np.seterr(all='warn')
    # do not enable wandb output
    os.environ["WANDB_SILENT"] = "true"
    # start training ==================================================================
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_name = '{}_{:04d}'.format('pointnet', args.expID)
    init_wandb(args, run_name)
    create_save_dirs(args)
    train_cls_pose(args)
