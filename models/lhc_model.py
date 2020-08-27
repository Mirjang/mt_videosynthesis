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

# def GGDown(ch, bn = False, weight_norm = None): 
#     return nn.Sequential(*[GResBlock(ch, ch, kernel_size=(3,3), downsample_factor=1, upsample_factor=1, bn=bn, weight_norm=weight_norm),
#     nn.ReLU(), 
#     GResBlock(ch, ch*2, kernel_size=(3,3), downsample_factor=2, bn=bn, weight_norm=weight_norm),
#     nn.ReLU()])

# def GGUp(ch, bn = False, weight_norm = None): 
#     return nn.Sequential(*[GResBlock(ch, ch, kernel_size=(3,3), downsample_factor=1, upsample_factor=1, bn=bn, weight_norm=weight_norm),
#     nn.ReLU(), 
#     GResBlock(ch, ch//2, kernel_size=(3,3), upsample_factor=2, bn=bn, weight_norm=weight_norm),
#     nn.ReLU()])

def GGDown(ch, bn = False, weight_norm = SpectralNorm): 
    return nn.Sequential(*[
    nn.Conv2d(ch, ch*2, kernel_size=(3,3), stride = 1, padding = 1),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    ]) 

def GGUp(ch, bn = False, weight_norm = SpectralNorm): 
    return nn.Sequential(*[
    nn.Conv2d(ch, ch//2, kernel_size=(3,3), stride = 1, padding = 1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    ]) 

class LHC(nn.Module):

    def __init__(self, latent_dim=32, ngf=32,npf = 32,steps_per_frame = 1, pd = 2, knn = 8, sample_kernel = "exp", nframes=48, debug= None):
        super().__init__()
        self.debug = debug
        self.latent_dim = latent_dim
        self.nframes = nframes -1 # first frame is just input frame
        self.knn = knn
        self.steps_per_frame = steps_per_frame
        encoder = []

        def conv_relu(in_dims, out_dims, stride = 1):
            layer = [nn.Conv2d(in_dims, out_dims, kernel_size=3, bias=True, padding=1, stride=stride, padding_mode="reflect")]
            layer += [nn.ReLU()]
            return layer

        encoder =[
            *conv_relu(3,ngf, stride = 1),
            GGDown(ngf), #-> ngf*2
            nn.ReLU(), 
         #   GGDown(ngf*2),  #-> ngf*4
         #   nn.ReLU(), 
            *conv_relu(ngf*2, npf, stride=1),
            nn.AdaptiveAvgPool2d((latent_dim,latent_dim))
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

        #self.distance_kernel = kernels.TruncatedExponentialKernel(sigma = 1., truncation_dist = .75)
        self.distance_kernel = kernels.LinearKernel()
        #self.distance_kernel = kernels.ExponentialKernel(sigma = 1)

        decoder = [
            nn.Upsample(scale_factor=2),
            *conv_relu(npf,ngf*2),
          #  GGUp(ngf*4), #-> ngf*2
          #  nn.ReLU(), 
            GGUp(ngf*2),  #-> ngf
            nn.ReLU(), 
            nn.Conv2d(ngf, 3, kernel_size=3, bias=True, padding=1, stride=1, padding_mode="reflect"),
        ]
        self.decoder = nn.Sequential(*decoder)

        self.it = 0

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
        x = x * 2 - 1
        if len(x.shape) == 5: # B x T x 3 x W x H -> B x 3 x W x H (first frame - other methods might use more frames -- esp for easier training)
            x = x[:,0,...]
        first_frame = x.unsqueeze(1)
        x = self.encoder(x)
        B, C, W, H = x.shape
        y = torch.zeros((self.nframes,B,C,W,H), device = x.device)
        num_particles = W*H
        kernel_scale = 1/math.sqrt(W**2 + H**2)
        #x_part = x.view(B, C, W*H)
        x_part = x.view(B, C, num_particles)
        ref_pos = self.uniform_grid(B,W,H,device=x.device) #B,2,W,H
        pos = self.uniform_grid(B,W,H,device=x.device).view(B,2, -1)


        
        for i in range(self.nframes):
        #    print(f"----------------{i}-----------")
        #    print("x: ", x_part.min().item(), x_part.max().item())#, v.min().item(), v.max().item())
        #    assert not (x_part != x_part).any(), f"x NAN at {i}"
        #    assert not (pos != pos).any(), f"pos NAN at {i}"

            for _ in range(self.steps_per_frame):
                # update based on neighbors
                if self.knn > 1:
                    d = (pos[...,None]@pos[...,None,:]).sum(dim=1)#outer product
                    _, knn_ind = torch.topk(d, self.knn, dim = 1, largest=False, sorted=False)
                    x_part = 1/self.knn * x_part[knn_ind].sum(dim = 1)

                # update latent info and compute velocity
                #x_part = self.rule_mlp(x_part)
                #x_part, v = xv.split([C, 2], dim = 1)# B,C+2,WxH --> B,C,WxH; B,2,WxH

                v = self.velocity_mlp(x_part)
                #integrate position
               # pos += v #.clamp(-1e5,1e5)


#            pos = pos.clamp(-1,1)
            #sample frame
            #distance between the reference pos (element in matrix) and actual pos of the particle
            #this is just kindof a hack so we can use std pytorch to simulate our particles
            #ideally we would so something like (differentiable) splatting here
            #but im lazy
           # print("v: ", v.min().item(), v.max().item())
            pos_e = pos.view(B, 2, W, H).unsqueeze(2).expand(B,2, num_particles, W, H)
            ref_pos_e = ref_pos.unsqueeze(2).expand(B,2, num_particles, W, H)
            #print(pos_e.shape, ref_pos_e.shape)
            dist = ((pos_e - ref_pos_e)**2).sum(dim = 1, keepdim = True) # B,1, WxH, W, H
            kdist = self.distance_kernel(dist, scale = kernel_scale)
           # pd = pos_e - ref_pos_e
           # print("p: " , pd.min().item(), pd.max().item(), (pd**2).min().item(), (pd**2).max().item())
           # print("d: ", dist.min().item(), dist.max().item(), "NAN" if (dist != dist).any() else "")
           # print("k: ", kdist.min().item(), kdist.max().item(), "NAN" if (kdist != kdist).any() else "")
            
            if self.debug: 
                for x in range (4): 
                    setattr(self.debug,f"heatmap_{x + 4*i}", (dist[:,:,x,...].detach().cpu() +1) /8 )
            #kdist = kdist.view(B, 1, num_particles)#.expand(B,1, num_particles, W, H) # B, 1 W*H
            #print(kdist.shape, x_part.shape)
            x_part_e = x_part.view(B, C, W, H).unsqueeze(2) # B, C, 1, W, H
            #print(x_part_e.shape)
            #print((kdist*x_part_e).shape)
            sample = torch.sum(kdist * x_part_e, dim = 2)#(B, 1, W*H, W, H) * (B, C, 1, W, H) -> (B, C, W, H)
            y[i] = sample

        y = y.permute(1,0,2,3,4).contiguous() #B,T,C,W,H
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
        return 'DHCModel'

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
        self.num_display_frames = min(opt.num_display_frames, int(opt.max_clip_length*opt.fps)-1)
        self.nframes = int(opt.max_clip_length * opt.fps / opt.skip_frames)
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

        #4,opt.niter + opt.niter_decay + 1 -opt.epoch_count
        self.activity_diag = torch.zeros((1,4))
        self.activity_diag_indices = {"Gs":0,"Gt":1, "Ds":2,"Dt":3}
        #self.activity_diag_colors = {"Gs":torch.tensor([1,0,0]),"Gt":torch.tensor([1,0,0]), "Ds":torch.tensor([0,1,0]),"Dt":torch.tensor([0,0,1])}
        self.activity_diag_colors = {"Gs":1,"Gt":1, "Ds":1,"Dt":1}

        self.activity_diag_plt = {"opts": {
                    'title': "Activity",
                    'legend': list(self.activity_diag_indices.keys()),
                    'xlabel': 'step',
                    'ylabel': 'active',
                    } ,
                    "Y":self.activity_diag,
                    "X":[0]
            }
        # self.lossperframe_plt = {"opts": {
        #             'title': "Loss per frame",
        #             #'legend': ["L1 loss per frame"],
        #             'xlabel': 'frame',
        #             'ylabel': 'loss',
        #             } ,
        #   #          "Y":np.array((1,1)), "X":np.array((1,1))
        #     }

        for i in range(self.num_display_frames):
            self.visual_names += [f"frame_{i}"]

        self.num_heatmaps = self.nframes * 4 -4
        for i in range(self.num_heatmaps):
            self.visual_names += [f"heatmap_{i}"]

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
        self.conditional = True
        self.wgan = False
        if not self.wgan:
            self.loss_names += ['accDs_real','accDs_fake','accDt_real', 'accDt_fake']
            self.visual_names.append("activity_diag_plt")

  
        netG = LHC(latent_dim=32, ngf=16,npf = 128,steps_per_frame = 1, pd = 2, knn = 0, sample_kernel = "exp", nframes=self.nframes, debug = self)

        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.conditional = False
            #default chn = 128
            netDs = DvdSpatialDiscriminator(chn = 8, sigmoid = not self.wgan, cgan = self.conditional)
            self.netDs = networks.init_net(netDs, opt.init_type, opt.init_gain, self.gpu_ids)

            #default chn = 128
            netDt = DvdTemporalDiscriminator(chn = 8, sigmoid = not self.wgan)
            self.netDt = networks.init_net(netDt, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(False).to(self.device)
            self.criterionWGAN = networks.WGANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            # self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), opt.lr)
            #self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
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

