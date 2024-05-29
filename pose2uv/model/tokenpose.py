# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
# import timm
import math
from .tokenpose_base import TokenPose_TB_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class TokenPose_T(nn.Module):

    def __init__(self):

        # extra = cfg.MODEL.EXTRA

        super(TokenPose_T, self).__init__()

        print('here is toknepose.py')
        ##################################################
        self.transformer = TokenPose_TB_base(feature_size=[256,192],patch_size=[16,12],
                                 num_keypoints =17, dim =192,
                                 channels=3,
                                 depth=12,heads=16,
                                 mlp_dim = 192*3,
                                 apply_init=True,
                                 hidden_heatmap_dim=64*48//8,
                                 heatmap_dim=64*48,
                                 heatmap_size=[64,48],
                                 pos_embedding_type='sine-full')
        ###################################################3

    def forward(self, data):
        x = self.transformer(data['img192256'])
        return dict(preheat=x)

    def init_weights(self, pretrained=''):
        pass


def tokenpose (is_train=True,INIT_WEIGHTS=True):
    model = TokenPose_T()
    if is_train and INIT_WEIGHTS:
        model.init_weights('hrnet_w32-36af842e.pth')

    return model
