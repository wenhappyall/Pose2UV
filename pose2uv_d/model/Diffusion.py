import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Model import UNet
from torchvision.utils import save_image
import os
import numpy as np
import cv2
from utils.imutils import *


modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000, # total step for model training
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4, # noise figure at the beginning
        "beta_T": 0.02, # noise figure at the end
        "img_size": 32, 
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "my_t": 50
        }
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T


        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double()) # betas ∈ (beta_1 , beta_T), number:T tensor, type: double()
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0) # multiply-accumulate by dim = 0, e.g. [0.9, 0.8, 0.7, 0.6, 0.5]——>[0.9, 0.72, 0.504, 0.3024, 0.1512]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    
    def forward(self, x_0, img):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device) # random step
        noise = torch.randn_like(x_0) # random noise and shape like x_0
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + # extract value from sqrt_alphas_bar by t and x_0 shape
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t, img), noise, reduction='none')
        return loss
        # t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device) # random step
        # noise = torch.randn_like(x_0) # random noise and shape like x_0
        # t = modelConfig["my_t"]
        # loop = int(modelConfig["T"] / t)    
        # for e in range(loop):
        #     t = torch.tensor(t,device=x_0.device)
        #     t = t.unsqueeze(0)
        #     noise = torch.randn_like(x_0)
        #     x_t = (
        #         extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + # extract value from sqrt_alphas_bar by t and x_0 shape
        #         extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)           
        #     x_t = x_t[0]
        #     x_t = x_t.detach().cpu().numpy().astype(np.float32)
        #     x_t = x_t.transpose(1, 2, 0)
        #     x_t = np.max(x_t, axis=2)
        #     x_t = convert_color(x_t*255)
        #     heatmap_name = "%05ddiffusion_heatmap.jpg"%(e)
        #     if not os.path.exists(modelConfig["sampled_dir"]):
        #         os.makedirs(modelConfig["sampled_dir"])
        #     cv2.imwrite(os.path.join(modelConfig["sampled_dir"], heatmap_name), x_t)
        #     t = int(t)+50
        # loss = F.mse_loss(self.model(x_t, t, img), noise, reduction='none')
        # return loss



class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T] # padding with value = 1，first column padding, second column no padding, take top T 

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, img):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t, img) # model = noisy predictor, predict noise
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T, img):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)): # time steps is sorted in reverse order
            print(time_step) 
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, img=img)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
       
class Diffusion(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.device = torch.device("cuda")
        self.net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(self.device)
        self.trainer  = GaussianDiffusionTrainer( self.net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])
        self.sampler = GaussianDiffusionSampler( self.net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])
    
    def forward(self, data): # Forward propagation, the process of inputting data and generating an output
        img, fullheat = data['img'], data['fullheatmap'] # data contains a dictionary of images and whole-body heatmaps
        if self.training :  
           loss =  self.trainer(fullheat[2], img)
           x_0 = None
        else:
            batchsize, num_joint, h, w = fullheat[2].shape[:]
            noisyImage = torch.randn(
            size=[ batchsize, num_joint, h, w], device=self.device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            # saveNoisy = saveNoisy[0]
            # saveNoisy = saveNoisy.detach().cpu().numpy().astype(np.float32)
            # saveNoisy = saveNoisy.transpose(1, 2, 0)
            # saveNoisy = np.max(saveNoisy, axis=2, keepdims=True)
            # saveNoisy = torch.tensor(saveNoisy)
            # noisy_name = "noisy7777.png"
            # save_image(saveNoisy, os.path.join(
            # modelConfig["sampled_dir"], noisy_name))
            x_0 = self.sampler(saveNoisy, img)
            x_0 = x_0 * 0.5 + 0.5
            loss = 0.
        return {'loss':loss, 'x_0':x_0}