import torch
import torch.nn as nn
from torch.nn import functional as F
from util.util import *
from .base_model import BaseModel
from . import networks
import numpy as np
import math
import functools
import random
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
from trajgru.trajgru import TrajGRU
from stylegan2.model import StyledConv
#import torch.nn.utils.spectral_norm as SpectralNorm

class UnconditionalModel(BaseModel):
    def name(self):
        return 'UnconditionalModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_S', type=float, default=.1, help='weight for spatial loss')
            parser.add_argument('--lambda_T', type=float, default=.1, help='weight for temporal loss')
            parser.add_argument('--lambda_GP', type=float, default=1, help='weight for gradient penalty')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.nframes = int(opt.max_clip_length * opt.fps / opt.skip_frames)
        self.num_display_frames = min(opt.num_display_frames, self.nframes - 1) //2 *2
        self.opt = opt
        self.iter = 1
        self.train_range = (1,self.nframes)
        self.in_dim = 120

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ["prediction_target_video"]#, "activity_diag_plt"]#, "lossperframe_plt"]
        for i in range(self.num_display_frames):
            self.visual_names += [f"frame_{i}"]
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Gs', 'Gt', 'Ds', 'Dt']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG', 'netDs', 'netDt']
        else:  # during test time, only load Gs
            self.model_names = ['netG']
        # load/define networks
        
        self.conditional = opt.conditional
        self.wgan = not opt.no_wgan
        if not self.wgan:
            self.loss_names += ['accDs_real','accDs_fake','accDt_real', 'accDt_fake']
        input_nc = opt.input_nc

        netG = DvdGenerator(in_dim=self.in_dim, latent_dim=4, n_class = 0, ch = opt.ch_g, n_frames = self.nframes, hierar_flag=False)
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.ndsframes = opt.dvd_spatial_frames
            assert self.nframes > self.ndsframes+1, "number of frames sampled for disc should be leq to number of total frames generated (length-1)"
       
            #default chn = 128
            netDs = DvdSpatialDiscriminator(chn = opt.ch_ds, sigmoid = not self.wgan, cgan = self.conditional)
            self.netDs = networks.init_net(netDs, opt.init_type, opt.init_gain, self.gpu_ids)

            #default chn = 128
            netDt = DvdTemporalDiscriminator(chn = opt.ch_dt, sigmoid = not self.wgan)
            self.netDt = networks.init_net(netDt, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(False).to(self.device)
            self.criterionWGAN = networks.WGANLoss(adv_loss = 'wgan-gp').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            # self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
           # self.optimizer_G = torch.optim.SGD(self.netG.parameters(), opt.lr)
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

    def set_input(self, input):
        self.target_video = input['VIDEO'].to(self.device).permute(0,1,4,2,3).float() / 255.0 #normalize to [0,1] for vis and stuff
        self.input = self.target_video[:,0,...]#first frame
        if self.opt.use_segmentation: 
            self.input = torch.cat([self.input, input["SEGMENTATION"].to(self.device)], dim = 1)
        self.noise_input = torch.empty((self.target_video.shape[0], self.in_dim)).normal_(mean=0, std=1)

        _, T, *_ = self.target_video.shape
        self.target_video = self.target_video[:, :min(T,self.nframes),...]

    def forward(self, frame_length = -1, train = True):
        if hasattr(self.netG, "nFrames"):
            self.netG.nFrames = frame_length if frame_length>0 else self.nframes
 
        self.predicted_video = self.netG(self.noise_input)
        print(self.predicted_video.shape, self.target_video.shape)
        if self.target_video.size(1) >= self.nframes: 
            self.prediction_target_video = torch.cat([self.predicted_video[:, :self.nframes,...].detach().cpu(), self.target_video.detach().cpu()], dim = 4)
        else: 
            self.prediction_target_video = self.predicted_video[:, :self.nframes,...].detach().cpu()

        for i in range(self.num_display_frames//2):
            setattr(self,f"frame_{i}", self.predicted_video[:,i,...].detach().cpu() )
        for i in range(self.num_display_frames//2):
            ith_last = self.num_display_frames//2 -i +1
            setattr(self,f"frame_{i + self.num_display_frames//2}", self.predicted_video[:,-ith_last,...].detach().cpu() )
       
    def epoch_frame_length(self, epoch):
        increase_intervals = 5
        iter_per_interval = 5
        return max(8,min(self.nframes // increase_intervals * (epoch // iter_per_interval +1), self.nframes))

    def gradient_penalty(self, real_img, fake_img, net):
        # Compute gradient penalty
        alpha = torch.rand(real_img.size(0), 1, 1, 1, 1).to(self.device).expand_as(real_img)
        interpolated = torch.tensor(alpha * real_img.data + (1 - alpha) * fake_img.data, requires_grad=True)
        out = net(interpolated)

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out.size()).to(fake_img.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        # Backward + Optimize
        return d_loss_gp

    def sample_frames(self, vid, detach = False):
        _, T, *_ = self.predicted_video.shape
        frames=[]
        for i in random.sample(range(1, T), min(T-1,self.ndsframes)):
            f = vid[:,i,...]
            if detach:
                f = f.detach().to(self.device)
            if self.conditional:
                f = torch.cat([self.input, f], dim = 1)
            frames.append(f)
        return torch.stack(frames, dim = 0)

    def backward_Ds_wgan(self, train_threshold = 0):
        fake = self.sample_frames(self.predicted_video, detach=True)
        real = self.sample_frames(self.target_video, detach=False)
        # Fake
        pred_fake = self.netDs(fake)
        self.loss_Ds_fake = torch.mean(pred_fake)
        # Real
        pred_real = self.netDs(real)
        self.loss_Ds_real = torch.mean(pred_real)
        self.loss_Ds_GP = self.gradient_penalty(real,fake,self.netDs)
        # Combined loss
        self.loss_Ds = (self.loss_Ds_real - self.loss_Ds_fake) + self.opt.lambda_GP * self.loss_Ds_GP
        return True

    def backward_Dt_wgan(self, train_threshold = 0):
        # Fake
        pred_fake = self.netDt(self.predicted_video.detach())
        self.loss_Dt_fake = torch.mean(pred_fake)
        # Real
        pred_real = self.netDt(self.target_video)
        self.loss_Dt_real = torch.mean(pred_real)
        self.loss_Dt_GP = self.gradient_penalty(self.target_video, self.predicted_video.detach(), net = self.netDt)
        # Combined loss
        self.loss_Dt = (self.loss_Dt_real - self.loss_Dt_fake) + self.opt.lambda_GP * self.loss_Dt_GP
        return True

    def backward_G_wgan(self, epoch, train_threshold = 0):
        self.loss_G_L1 = 0
        fake_frames = self.sample_frames(self.predicted_video, detach=False)
        pred_fake_ds = self.netDs(fake_frames)
        self.loss_Gs = torch.mean(pred_fake_ds)
        pred_fake_dt = self.netDt(self.predicted_video)
        self.loss_Gt = torch.mean(pred_fake_dt)

        self.loss_Gaux = getattr(self.netG.module, "L_aux", 0)

        self.loss_G = self.loss_Gs * self.opt.lambda_S\
            + self.loss_Gt * self.opt.lambda_T\
            + self.loss_Gaux * self.opt.lambda_AUX
        if epoch <= self.opt.pretrain_epochs:
            self.loss_G_L1 =self.criterionL1(self.predicted_video, self.target_video) * self.opt.lambda_L1
            self.loss_G += self.loss_G_L1
        return True

    def compute_losses(self, epoch, verbose = False):
        verbose = verbose or self.opt.verbose
        Te = self.nframes
        B, TT,*_ = self.target_video.shape

        self.forward(frame_length=Te)
        _, T,*_ = self.predicted_video.shape
        T = min(T,TT,Te) # just making sure to cut target if we didnt predict all the frames and to cut prediction, if we predicted more than target (i.e. we already messed up somewhere)

        # update Generator every n_critic steps
        if self.iter % self.opt.n_critic == 0:
            self.set_requires_grad(self.netDs, False)
            self.set_requires_grad(self.netDt, False)
            self.set_requires_grad(self.netG, True)
            update_G = self.backward_G_wgan(epoch, train_threshold = self.opt.tlg)
            if update_G: 
                self.loss_G = self.loss_G / self.opt.n_acc_batches
                self.loss_G.backward()            
            if verbose:
                diagnose_network(self.netG, "netG")
        else: 
            #update Discriminator(s)
            self.set_requires_grad(self.netDs, True)
            update_Ds = self.backward_Ds_wgan()
            if update_Ds: 
                self.loss_Ds = self.loss_Ds / self.opt.n_acc_batches
                self.loss_Ds.backward()
            if verbose:
                diagnose_network(self.netDs,"Ds")

            self.set_requires_grad(self.netDt, True)
            update_Dt = self.backward_Dt_wgan()
            if update_Dt: 
                self.loss_Dt = self.loss_Dt / self.opt.n_acc_batches
                self.loss_Dt.backward()
            if verbose:
                diagnose_network(self.netDt,"Dt")

    def optimize_parameters(self, epoch, verbose = False):
        # update Generator every n_critic steps
        if self.iter % self.opt.n_critic == 0:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.clip_grads)
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
        else: 
            nn.utils.clip_grad_norm_(self.netDs.parameters(), self.opt.clip_grads)
            self.optimizer_Ds.step()
            self.optimizer_Ds.zero_grad()
            nn.utils.clip_grad_norm_(self.netDt.parameters(), self.opt.clip_grads)
            self.optimizer_Dt.step()
            self.optimizer_Dt.zero_grad()
        self.iter += 1

    def compute_validation_losses(self, secs = 1, fps = 30):
        T = secs * fps *1.0
        with torch.no_grad():
            self.netG.eval()
            self.forward(frame_length=T, train=False)
            self.netG.train()
            _, T,*_ = self.predicted_video.shape
            _, TT,*_ = self.target_video.shape

            # print(torch.min(self.predicted_video), torch.max(self.predicted_video))
            # print(torch.min(self.target_video), torch.max(self.target_video))
            T = min(T,TT) # just making sure to cut target if we didnt predict all the frames and to cut prediction, if we predicted more than target (i.e. we already messed up somewhere)
            loss_L1 = self.criterionL1(self.predicted_video, self.target_video) * self.opt.lambda_L1

            ds_perframe = {"opts": {
                    'title': "Loss per frame",
                    'legend': ["Ds prediction"],
                    'xlabel': 'frame',
                    'ylabel': 'loss',
                    } ,
            }
            dspf = []
            for i in range(T):
                fake = self.predicted_video[:,i,...].detach().to(self.device)
                if self.conditional:
                    fake = torch.cat([self.input, fake], dim = 1)
                dspf.append(torch.mean(self.netDs(fake)).item())

            ds_perframe["Y"] = dspf
            ds_perframe["X"] =list(range(len(dspf)))
            dt_fake = torch.mean(self.netDt(self.predicted_video)).item()

            return OrderedDict([("val_l1", loss_L1.item()), ("val_Ds", sum(dspf)/len(dspf)), ("val_Dt", dt_fake)]), \
                    OrderedDict([("val_pred_tar_video", self.prediction_target_video),("val_ds_per_frame_plt", ds_perframe), ])