'''
 @FileName    : dataset.py
 @EditTime    : 2022-09-27 16:03:55
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
from random import random
import torch
import numpy as np
import cv2
import constants
from datasets.base import base
from utils.imutils import convert_color
from utils.dataset_handle import create_UV_maps, create_poseseg

class PoseSegData(base):
    def __init__(self, train=True, data_folder='', name='', smpl=None, occlusions=None, uv_generator=None, dtype=torch.float32):
        super(PoseSegData, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.occlusions = occlusions
        self.generator = uv_generator

        self.lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        self.halpe_to_lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,0]
        self.halpe_to_coco = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

        if self.is_train:
            dataset_annot = os.path.join(self.dataset_dir, 'annot/test.pkl')
            self.eval = False
        else:
            self.eval = True
            dataset_annot = os.path.join(self.dataset_dir, 'annot/test.pkl')
        params = self.load_pkl(dataset_annot)

        self.pose2ds, self.pose2ds_pred, self.imnames, self.masks, self.img_size, self.bboxs = [], [], [], [], [], []
        for seq in params:
            if len(seq) < 1:
                continue
            for i, frame in enumerate(seq):
                for key in frame.keys():
                    if key in ['img_path', 'h_w']:
                        continue
                    self.img_size.append(frame['h_w'])
                    self.pose2ds.append(np.array(frame[key]['coco_joints_2d'], dtype=self.np_type).reshape(-1, 3))
                    self.pose2ds_pred.append(np.array(frame[key]['halpe_joints_2d_pred'], dtype=self.np_type).reshape(-1,3))
                    self.imnames.append(frame['img_path'])
                    self.bboxs.append(np.array(frame[key]['bbox'], dtype=self.np_type).reshape(-1,))
                    self.masks.append(frame[key]['segmentation'])
                    
        del frame
        del params

        self.len = len(self.img_size)

    def vis_input(self, image, mask, pred_heatmap, gt_heatmap):
        # Show image
        image = image.detach().cpu().numpy().transpose((1,2,0))[:,:,::-1]
        self.vis_img('img', image)

        # Show mask
        mask = mask * 255
        self.vis_img('mask', mask)

        # Show gt heatmap
        gt_heatmap = np.max(gt_heatmap, axis=0)
        gt_heatmap = convert_color(gt_heatmap*255)
        gt_heatmap = cv2.addWeighted(gt_heatmap, 0.5, (image.copy()*255).astype(np.uint8), 0.5, 0)
        self.vis_img('gt_heatmap', gt_heatmap)

        # Show pred heatmap
        pred_heatmap = np.max(pred_heatmap, axis=0)
        pred_heatmap = convert_color(pred_heatmap*255)
        pred_heatmap = cv2.addWeighted(pred_heatmap, 0.5, (image.copy()*255).astype(np.uint8), 0.5, 0)
        self.vis_img('pred_heatmap', pred_heatmap)

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        gt_input = 0
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
            if np.random.uniform() <= 0.5:
                gt_input = 1

        return flip, pn, rot, sc, gt_input

    # Data preprocess
    def create_data(self, index=0):
        
        # load data
        image_path = os.path.join(self.dataset_dir, self.imnames[index].replace('\\', '/')) 
        img_id = int(os.path.basename(self.imnames[index].replace('\\', '/')).split('.')[0])
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        mask = self.masks[index] # original mask is list
        mask = self.annToMask(mask, img_h, img_w) # convert to tensor

        bbox = self.bboxs[index].reshape(-1, 2) # -1: infer the length of the dimension. sum(element) is same,2: The second dimension of the output contain two elements [1,2,3,4]——>[[1,2],[3.4]]
        lt = np.array(bbox[0]) # get the coordinates of the upper-left corner.
        rb = np.array(bbox[1])
        # select 2D pose
        if self.pose2ds_pred[index] is not None:
            input_kp_2d = self.pose2ds_pred[index][self.halpe_to_coco]
        else:
            input_kp_2d = np.zeros((17, 3), dtype=self.np_type)
        # 13 keypoints, each keypoint consists of x, y coordinates and confidence value
        assert input_kp_2d.shape == (17, 3)# and kp_2d[:,2].max() <= 1.5

        gt_kp_2d = self.pose2ds[index].copy()

        data = create_poseseg(img, mask, lt, rb, input_kp_2d, gt_kp_2d, self.smpl, self.generator, image_path, occlusions=self.occlusions, is_train=self.is_train)
        data['img_id'] = img_id
        return data


    def __getitem__(self, index):

        data = self.create_data(index)

        return data

    def __len__(self):
        return self.len













