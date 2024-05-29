from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn

from model.posenet import posenet
from model.segnet import segnet
from model.tokenpose import TokenPose_T


class Res_catconv(nn.Module): # The definition class inherits from the nn.Module class and is used to build the neural network blocks
     # to initialize an instance of the Res_catconv class, 
     # two parameters segnet for the neural network model for the segmentation task, posenet for the neural network model for the pose estimation task
    def __init__(self, segnet, posenet):
        super(Res_catconv, self).__init__()
        self.segnet = segnet
        self.posenet = posenet

    def forward(self, data): # Forward propagation, the process of inputting data and generating an output
        img, fullheat = data['img'], data['fullheat'] # data contains a dictionary of images and whole-body heatmaps
        partialheat = self.posenet(img, fullheat[0])
        pre_mask = self.segnet(img, partialheat) 

        return dict(encoded=None, decoded=None, mask=pre_mask[-1], heatmap=fullheat[-1], premask=pre_mask, preheat=partialheat)
    # 1. extract image(img)and heatmap (fullheat) from the data dictionary
    # 2. Process the image and the initial heatmap using posenet model to get partialheat
    # 3. Process the img and partialheat using segnet to get the predicted mask pre_mask
    # 4. Returns a dictionary with encoding, decoding, mask, heatmap, prediction mask, and partial heatmap information

# create network
def poseseg(generator=None): # to create a Res_catconv model instance.
    pose = posenet() # Create a posenet model instance.
    seg = segnet() # Create the segnet model instance.
    UV_net = Res_catconv(seg, pose) # Build a Res_catconv model instance using posenet and segnet instances.
    return UV_net # Return the created Res_catconv model instance.
