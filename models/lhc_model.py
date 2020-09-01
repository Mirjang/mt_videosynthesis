import torch
import torch.nn as nn
from torch.nn import functional as F
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel
from . import networks
import numpy as np
import math
import functools
import random
from .networks import VGG16, ConvLSTMCell, ConvGRUCell
from collections import OrderedDict, namedtuple
from torchvision import models
#from .dvdgan.GResBlock import *
from dvdgan.Discriminators import SpatialDiscriminator as DvdSpatialDiscriminator
from dvdgan.Discriminators import TemporalDiscriminator as DvdTemporalDiscriminator
from dvdgan.Generator import Generator as DvdGenerator
from dvdgan.GResBlock import GResBlock
from dvdgan.ConvGRU import ConvGRU
from dvdgan.Normalization import SpectralNorm
#def SpectralNorm(x):
#    return x
from .dvdgansimple_model import GRUEncoderDecoderNet
import models.kernels as kernels
#import torch.nn.utils.spectral_norm as SpectralNorm

def GGDown(ch, bn = False, weight_norm = SpectralNorm): 
    return [GResBlock(ch, ch, kernel_size=(3,3), downsample_factor=1, upsample_factor=1, bn=bn, weight_norm=weight_norm),
    nn.ReLU(), 
    GResBlock(ch, ch*2, kernel_size=(3,3), downsample_factor=2, bn=bn, weight_norm=weight_norm),
    nn.ReLU()]

def GGUp(ch, bn = False, weight_norm = SpectralNorm): 
    return [GResBlock(ch, ch, kernel_size=(3,3), downsample_factor=1, upsample_factor=1, bn=bn, weight_norm=weight_norm),
    nn.ReLU(), 
    GResBlock(ch, ch//2, kernel_size=(3,3), upsample_factor=2, bn=bn, weight_norm=weight_norm),
    nn.ReLU()]

# def GGDown(ch, bn = False, weight_norm = SpectralNorm): 
#     return [
#     nn.Conv2d(ch, ch*2, kernel_size=(3,3), stride = 1, padding = 1),
#     nn.ReLU(),
#     nn.AvgPool2d(2,2),
#     ] 

# def GGUp(ch, bn = False, weight_norm = SpectralNorm): 
#     return [
#     nn.Conv2d(ch, ch//2, kernel_size=(3,3), stride = 1, padding = 1),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2),
#     ] 
def conv_relu(in_dims, out_dims, stride = 1, weight_norm = SpectralNorm):
    layer = [weight_norm(nn.Conv2d(in_dims, out_dims, kernel_size=3, bias=True, padding=1, stride=stride, padding_mode="reflect"))]
    layer += [nn.ReLU()]
    return layer

class LHC(nn.Module):

    def __init__(self, latent_dim=32, ngf=32,npf = 32,steps_per_frame = 1, pd = 2, knn = 8, sample_kernel = "exp", nframes=48, debug= None):
        super().__init__()
        self.debug = debug
        self.latent_dim = latent_dim
        self.nframes = nframes -1 # first frame is just input frame
        self.knn = knn
        self.steps_per_frame = steps_per_frame
        self.truncation_dist = .8

        encoder =[
            *conv_relu(3,ngf),
            *GGDown(ngf), #-> ngf*2
            nn.ReLU(), 
            *GGDown(ngf*2),  #-> ngf*4
            *conv_relu(ngf*4,npf),
        ]
        self.encoder = nn.Sequential(*encoder)

        rule_mlp = [
            nn.Conv1d(npf, npf//2, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(npf//2, npf, kernel_size=1, bias=True),
            nn.ReLU(),
        ]
        self.rule_mlp = nn.Sequential(*rule_mlp)

        velocity_mlp = [
            nn.Conv1d(npf, npf//2, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(npf//2, 2, kernel_size=1, bias=True),
            nn.Tanh()
        ]
        self.velocity_mlp = nn.Sequential(*velocity_mlp)

        self.distance_kernel = kernels.TruncatedExponentialKernel(sigma = 1, truncation_dist = self.truncation_dist)
        #self.distance_kernel = kernels.LinearKernel()
        #self.distance_kernel = kernels.ExponentialKernel(sigma = 1)

        decoder = [
            nn.Upsample(scale_factor=2),
            *conv_relu(npf,ngf*2),
          #  GGUp(ngf*4), #-> ngf*2
          #  nn.ReLU(), 
            *GGUp(ngf*2),  #-> ngf
            nn.ReLU(), 
            nn.Conv2d(ngf, 3, kernel_size=3, bias=True, padding=1, stride=1, padding_mode="reflect"),
        ]
        self.decoder = nn.Sequential(*decoder)

        # self.encoder = nn.Sequential(*conv_relu(3,4))
        # self.decoder = nn.Sequential(*conv_relu(4,3))

        self.it = 0

        if self.debug: 
            self.debug.visual_names.insert(0,"heatmap_video")

    def uniform_grid(self,B,W,H, device = None):
        step_h = 2./(H-1)
        step_w = 2./(W-1)
        py = torch.arange(start=-1, end=1 + step_h, step=step_h, device=device)
        px = torch.arange(start=-1, end=1 + step_w, step=step_w, device=device)
        assert px.shape[0] == W and py.shape[0] == H
        px = px.unsqueeze(1).expand(1,W,H)
        py = py.unsqueeze(0).expand(1,W,H)
        return torch.cat([px,py], dim = 0).expand(B,-1,-1,-1).contiguous()

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        x = x * 2 - 1
        if len(x.shape) == 5: # B x T x 3 x W x H -> B x 3 x W x H (first frame - other methods might use more frames -- esp for easier training)
            x = x[:,0,...]
        first_frame = x.unsqueeze(1)

        x = self.encoder(x)
        B, C, W, H = x.shape
        y = torch.zeros((B,self.nframes,C,W,H), device = x.device)
        num_particles = W*H
        kernel_scale =1# 1/num_particles

        x_part = x.view(B, C, num_particles)
        ref_pos = self.uniform_grid(B,W,H,device=x.device) #B,2,W,H
        pos = self.uniform_grid(B,W,H,device=x.device).view(B,2, -1) #B,2,W*H

     #   y = torch.zeros((B,self.nframes,3,64,64), device = x.device)
 
        if self.debug: 
            self.debug.heatmap_video = torch.zeros((B,self.nframes,3,W,H))
            


        for i in range(self.nframes):
        #    print(f"----------------{i}-----------")
        #    print("x: ", x_part.min().item(), x_part.max().item())#, v.min().item(), v.max().item())
        #    assert not (x_part != x_part).any(), f"x NAN at {i}"
        #    assert not (pos != pos).any(), f"pos NAN at {i}"

            for _ in range(self.steps_per_frame):
                # update based on neighbors
                # if self.knn > 1:
                #     d = (pos[...,None]@pos[...,None,:]).sum(dim=1)#outer product
                #     _, knn_ind = torch.topk(d, self.knn, dim = 1, largest=False, sorted=False)
                #     x_neigh = 1/self.knn * x_part[knn_ind].sum(dim = 1)
                #x_part, v = xv.split([C, 2], dim = 1)# B,C+2,WxH --> B,C,WxH; B,2,WxH

                # update latent info and compute velocity
                x_part = x_part + self.rule_mlp(x_part)

                v = self.velocity_mlp(x_part)
                #integrate position
                pos = pos + v #.clamp(-1e-2,1e-2)
                pass


            pos = pos.clamp(-1.1,1.1) #keep stuff from going super bad while still allowing particles to go out of frame
            #sample frame
            #distance between the reference pos (element in matrix) and actual pos of the particle
            #this is just kindof a hack so we can use std pytorch to simulate our particles

            pos_e = pos.unsqueeze(-1).unsqueeze(-1) #B,2,W*H -> # B,2, WxH, 1,1
            ref_pos_e = ref_pos.unsqueeze(2).expand(B,2, num_particles, W, H)
            dist = ((pos_e - ref_pos_e)**2).sum(dim = 1, keepdim = True) # B,1, WxH, W, H
            kdist = self.distance_kernel(dist, scale = kernel_scale)

            x_part_e = x_part.view(B, C, W, H).unsqueeze(2) # B, C, 1, W, H
            sample = torch.sum(kdist * x_part_e, dim = 2)#(B, 1, W*H, W, H) * (B, C, 1, W, H) -> (B, C, W, H)
            hits = torch.sum(kdist > self.truncation_dist, dim = 2).detach()
            hits[hits == 0] = 1 #otherwise we´d have 0/0
            sample = sample / hits.float()
            #sample = x
            y[:,i] = sample

            if self.debug: 
                pass

                #print("x: ", x_part.min().item(), x_part.max().item())
                # print("v: ", v.min().item(), v.max().item())
                # print("p: ", pos.min().item(), pos.max().item())
                # print("r: ", ref_pos.min().item(), ref_pos.max().item())
                # print("d: ", dist.min().item(), dist.max().item(), "NAN" if (dist != dist).any() else "")
                print("k: ", kdist.min().item(), kdist.max().item(), "NAN" if (kdist != kdist).any() else "")
                hmax = 100
                self.debug.heatmap_video[:,i] = (torch.clamp(hits.float(), max = hmax)/hmax).detach().cpu().expand(-1, 3, -1,-1)
                if self.debug.iter % 200: 
                    print("hits: ", hits.min().item(), hits.max().item())


                # for ii in range(4): 
                #     debframe = [0, num_particles//2,num_particles//2+1, W*H -1][ii]
                #     setattr(self.debug,f"heatmap_{ii + 4*i}", (kdist[:,:,debframe,...].detach().cpu()))
                # if i <5: 
                #     smin,smax = sample.min(), sample.max()
                #     setattr(self.debug,f"heatmap_{2*i}", ((sample[:,:3,...] -smin) / (smax-smin)).detach().cpu())

                #     dsample = torch.tanh(self.decoder(sample))
                #     dsmin,dsmax = dsample.min(), dsample.max()
                #     setattr(self.debug,f"heatmap_{2*i +1}", ((dsample[:,:3,...] -dsmin) / (dsmax-dsmin)).detach().cpu())
                #     # if i == 0: 
                #     #     print("y: ",dsmin.item(), dsmax.item())

        y = y.view(-1, C, W, H)
        y = torch.tanh(self.decoder(y))
        *_, W, H = first_frame.shape
        y = y.view(B, self.nframes, 3, W, H) # B, T/S, C*S, W, H
        y = torch.cat([first_frame, y], dim = 1)

        y = (y+1) / 2.0 #[-1,1] -> [0,1] for vis
        return y

from .dvdgan_model import DvdGanModel
class LHCModel(DvdGanModel):
    def name(self):
        return 'LHCModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_S', type=float, default=.1, help='weight for spatial loss')
            parser.add_argument('--lambda_T', type=float, default=.1, help='weight for temporal loss')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for pretrain L1 loss')
            parser.add_argument('--lambda_GP', type=float, default=1, help='gradient penalty')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.nframes = int(opt.max_clip_length * opt.fps / opt.skip_frames)
        self.num_display_frames = min(opt.num_display_frames, self.nframes)

        self.opt = opt
        self.iter = 0
        self.ndsframes = opt.dvd_spatial_frames
        self.parallell_batch_size = opt.parallell_batch_size if opt.parallell_batch_size > 0 else opt.batch_size

        assert self.nframes > self.ndsframes+1, "number of frames sampled for disc should be leq to number of total frames generated (length-1)"
        print(self.device)

        self.train_range = (1,self.nframes)
        if self.opt.train_mode is "frame":
            self.train_range = (1,2)
        elif self.opt.train_mode is "video":
            self.train_range = (self.nframes-1,self.nframes)
        elif self.opt.train_mode is "mixed":
            self.train_range = (1,self.nframes)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ["prediction_target_video"]#, "activity_diag_plt"]#, "lossperframe_plt"]

        for i in range(self.num_display_frames):
            self.visual_names += [f"frame_{i}"]

        # self.num_heatmaps = self.nframes * 4 -4
        # for i in range(self.num_heatmaps):
        #     self.visual_names += [f"heatmap_{i}"]

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Gs', 'Gt', 'Ds', 'Dt']
        if opt.pretrain_epochs > 0:
            self.loss_names.append('G_L1')
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG', 'netDs', 'netDt']
        else:  # during test time, only load Gs
            self.model_names = ['netG']

        # load/define networks
        #default ch = 32
        self.in_dim = 1
        self.condition_gen = True
        self.conditional = False
        self.wgan = True
        if not self.wgan:
            self.loss_names += ['accDs_real','accDs_fake','accDt_real', 'accDt_fake']
        else: 
            self.loss_names += ['Ds_real', 'Ds_fake', 'Dt_real', 'Dt_fake', 'Ds_GP', 'Dt_GP']

  
        #debug
        netG = LHC(latent_dim=16, ngf=32,npf = 128,steps_per_frame = 1, pd = 2, knn = 0, sample_kernel = "exp", nframes=self.nframes, debug = self)
        #real
        #netG = LHC(latent_dim=16, ngf=32,npf = 128,steps_per_frame = 1, pd = 2, knn = 0, sample_kernel = "exp", nframes=self.nframes)

        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            #default chn = 128
            netDs = DvdSpatialDiscriminator(chn = 16, sigmoid = not self.wgan, cgan = self.conditional)
            self.netDs = networks.init_net(netDs, opt.init_type, opt.init_gain, self.gpu_ids)

            #default chn = 128
            netDt = DvdTemporalDiscriminator(chn = 16, sigmoid = not self.wgan)
            self.netDt = networks.init_net(netDt, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(False).to(self.device)
            self.criterionWGAN = networks.WGANLoss(adv_loss = 'wgan-gp').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            # self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            #self.optimizer_G = torch.optim.SGD(self.netG.parameters(), opt.lr)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_Ds = torch.optim.Adam(self.netDs.parameters(), lr=opt.lr * 1, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Ds)

            self.optimizer_Dt = torch.optim.Adam(self.netDt.parameters(), lr=opt.lr * 1, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Dt)
        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0
        self.loss_Ds = 0
        self.loss_Dt = 0
        self.loss_accDs_fake = 0
        self.loss_accDs_real = 0
        self.loss_accDt_real = 0 
        self.loss_accDt_fake = 0
        self.loss_Ds_fake = 0
        self.loss_Ds_real = 0
        self.loss_Dt_fake = 0
        self.loss_Dt_real = 0
        self.loss_Ds_GP = 0
        self.loss_Dt_GP = 0

    # def optimize_parameters(self, epoch, verbose = False):
    #     self.optimizer_G.step()
    #     self.optimizer_G.zero_grad()
    #     pass

    # def compute_losses(self, epoch, verbose = False):
    #     Te = self.nframes
    #     B, TT,*_ = self.target_video.shape

    #     self.forward(frame_length=Te)
    #     self.loss_G_L1 =self.criterionL1(self.predicted_video[:,1], self.target_video[:,1]) * self.opt.lambda_L1
    #     self.loss_G_L1.backward()
    #     #diagnose_network(self.netG,"G")
