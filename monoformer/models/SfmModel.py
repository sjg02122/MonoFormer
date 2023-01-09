# Copyright 2020 Toyota Research Institute.  All rights reserved.

from os import defpath
import random
from numpy.lib.type_check import imag
import torch.nn as nn
from wandb import wandb_agent
from monoformer.networks import depth
from monoformer.utils.image import flip_model, interpolate_scales
from monoformer.geometry.pose import Pose
from monoformer.utils.misc import make_list, visualize_attention
import numpy as np
import torch.nn.functional as F
import torch
import wandb
import cv2

        

class SfmModel(nn.Module):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 **kwargs):
        super().__init__()
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self._logs = {}
        self._losses = {}


        self._network_requirements = {
                'depth_net': True,  # Depth network required
                'pose_net': True,   # Pose network required
            }
        self._train_requirements = {
                'gt_depth': False,  # No ground-truth depth required
                'gt_pose': False,   # No ground-truth pose required
            }

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def losses(self):
        """Return metrics."""
        return self._losses

    def add_loss(self, key, val):
        """Add a new loss to the dictionary and detaches it."""
        self._losses[key] = val.detach()

    @property
    def network_requirements(self):
        """
        Networks required to run the model

        Returns
        -------
        requirements : dict
            depth_net : bool
                Whether a depth network is required by the model
            pose_net : bool
                Whether a depth network is required by the model
        """
        return self._network_requirements

    @property
    def train_requirements(self):
        """
        Information required by the model at training stage

        Returns
        -------
        requirements : dict
            gt_depth : bool
                Whether ground truth depth is required by the model at training time
            gt_pose : bool
                Whether ground truth pose is required by the model at training time
        """
        return self._train_requirements

    def add_depth_net(self, depth_net):
        """Add a depth network to the model"""
        self.depth_net = depth_net
        # from torchsummary import summary
        # summary(self.depth_net.cuda(),(3,640,192))

    def add_pose_net(self, pose_net):
        """Add a pose network to the model"""
        self.pose_net = pose_net



    def compute_inv_depths(self, image,file):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flip_lr = random.random() < self.flip_lr_prob if self.training else False
        inv_depths = []
        ouput_depth = flip_model(self.depth_net,image,flip_lr)
        ouput_depth = torch.unsqueeze(ouput_depth,1)
        inv_depths.append(ouput_depth)
        # for i in range(ouput_depth.shape[0]):
        #     wandb.log({str(file[i]):wandb.Image(ouput_depth[i])})        
        return inv_depths

    def compute_poses(self, image, contexts):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(image, contexts)
        # print(type(image),type(contexts))
        # from torchsummary import summary
        # summary(self.pose_net.cuda(),(image,contexts))
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def forward(self, batch, return_logs=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored

        Returns
        -------
        output : dict
            Dictionary containing predicted inverse depth maps and poses
        """
        idx_list = batch['idx'].tolist()
        inv_depths = self.compute_inv_depths(batch['rgb'],idx_list)
        # Generate pose predictions if available
        pose = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose = self.compute_poses(batch['rgb'],batch['rgb_context'])
        # Return output dictionary
        return {
            'inv_depths': inv_depths,
            'poses': pose,
        }

