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
from torchvision import models, transforms, utils
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
#import torch.nn.utils.spectral_norm as SpectralNorm
from stylegan2.model import Generator as Stylegan2Generator
from stylegan2.model import Discriminator as Stylegan2Discriminator

from stylegan2.non_leaking import augment
from stylegan2.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class StyleganModel(BaseModel):
    def name(self):
        return 'StyleganModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_S', type=float, default=.1, help='weight for spatial loss')
            parser.add_argument('--lambda_T', type=float, default=.1, help='weight for temporal loss')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for pretrain L1 loss')
            parser.add_argument('--lambda_GP', type=float, default=1, help='weight for gradient penalty')
            parser.add_argument('--lambda_AUX', type=float, default=0, help='weight for aux loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.nframes = int(opt.max_clip_length * opt.fps / opt.skip_frames)
        self.opt = opt
        self.iter = 1
        self.num_display_frames = min(16, opt.parallell_batch_size)
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = []#, "activity_diag_plt"]#, "lossperframe_plt"]
        for i in range(self.num_display_frames):
            self.visual_names += [f"frame_{i}"]
        for i in range(self.num_display_frames):
            self.visual_names += [f"input_{i}"]    
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'Ds']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG', 'netDs']
        else:  # during test time, only load Gs
            self.model_names = ['netG']
        # load/define networks
        
        self.latent = 512
        self.wgan = True

        self.accum = 0.5 ** (32 / (10 * 1000))
        self.ada_augment = torch.tensor([0.0, 0.0], device=self.device)
        self.ada_aug_p = .5
        self.ada_aug_step = .6 / (500 * 1000)
        self.ada_target = .6
        self.r_t_stat = 0
        self.augment_p = 0 

        self.loss_names += ['Ds_real', 'Ds_fake', 'r1', 'path']

        self.mean_path_length = 0
        self.augment = False
        self.reg_freq = 4
        self.mixing = .9
        self.loss_r1 = torch.tensor(0.0, device=self.device)
        self.loss_path = torch.tensor(0.0, device=self.device)
        self.path_lengths = torch.tensor(0.0, device=self.device)
        self.mean_path_length_avg = 0

        netG = Stylegan2Generator(opt.resolution, self.latent, 8, channel_multiplier=1 )
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.opt.lambda_AUX > 0: # and hasattr(self.netG, "L_aux"): 
            self.loss_names += ["Gaux"]
            self.loss_Gaux = 0

        if self.isTrain:
                   
            #default chn = 128
           # netDs = DvdSpatialDiscriminator(chn = 32, sigmoid = not self.wgan, cgan = False)
            netDs = Stylegan2Discriminator(opt.resolution, channel_multiplier=1)
            self.netDs = networks.init_net(netDs, opt.init_type, opt.init_gain, self.gpu_ids)

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

        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0
        self.loss_Ds = 0
        self.loss_Dt = 0
        self.loss_accDs_fake = 0
        self.loss_accDs_real = 0
        self.loss_Ds_fake = 0
        self.loss_Ds_real = 0
        self.loss_Ds_GP = 0

    def set_input(self, input):
        self.target_video = input['VIDEO'].to(self.device).permute(0,1,4,2,3).float() / 255.0 #normalize to [0,1] for vis and stuff
        self.input = self.target_video[:,0,...]#first frame
        if self.augment: 
            self.input, _ = augment(self.input, self.ada_aug_p)

        _, T, *_ = self.target_video.shape
        self.target_video = self.target_video[:, :min(T,self.nframes),...]

    def forward(self, frame_length = -1, train = True):
        noise = mixing_noise(self.input.size(0), self.latent, self.mixing, self.device)

        self.fake, _= self.netG(noise)
      #  self.fake = (torch.tanh(self.fake) +1) / 2
      #  print(self.fake.min(), self.fake.max())
        for i in range(self.num_display_frames):
            setattr(self,f"frame_{i}", self.fake[i:i+1,...].detach().cpu() )
        for i in range(self.num_display_frames):
            setattr(self,f"input_{i}", self.input[i:i+1,...].detach().cpu() )
        if self.augment: 
            self.fake,_ = augment(self.fake, self.ada_aug_p)
     

    def epoch_frame_length(self, epoch):
        increase_intervals = 5
        iter_per_interval = 5
        return max(8,min(self.nframes // increase_intervals * (epoch // iter_per_interval +1), self.nframes))

    def gradient_penalty(self, real_img, fake_img, net):
        # Compute gradient penalty
        alpha = torch.rand(real_img.size(0), 1, 1, 1).to(self.device).expand_as(real_img)
        interpolated = torch.tensor(alpha * real_img.data + (1 - alpha) * fake_img.data, requires_grad=True)
        out = net(interpolated)


        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        # Backward + Optimize
        return d_loss_gp

    def backward_Ds_wgan(self, train_threshold = 0):
        fake = self.fake.detach()

        real = self.input
        # Fake
        pred_fake = self.netDs(fake)
        self.loss_Ds_fake = torch.mean(pred_fake)
        # Real
        pred_real = self.netDs(real)

        self.loss_Ds_real = torch.mean(pred_real)

        #self.loss_Ds = d_logistic_loss(pred_real, pred_fake)
     #   self.loss_Ds_GP = self.gradient_penalty(real,fake,self.netDs)
        # Combined loss
        self.loss_Ds = (self.loss_Ds_real - self.loss_Ds_fake)# + self.opt.lambda_GP * self.loss_Ds_GP
        return True


    def backward_G_wgan(self, epoch):
        fake_frames = self.fake
        pred_fake_ds = self.netDs(fake_frames)

        #self.loss_Gs = g_nonsaturating_loss(pred_fake_ds)
        self.loss_Gs = torch.mean(pred_fake_ds)
        self.loss_G = self.loss_Gs * self.opt.lambda_S
        return True


    def compute_losses(self, epoch, verbose = False):
        verbose = verbose or self.opt.verbose

        self.forward()
        #update Discriminator(s)
        self.set_requires_grad(self.netDs, True)
        update_Ds = self.backward_Ds_wgan()# if self.wgan else self.backward_Ds(train_threshold = self.opt.tld)
        if update_Ds: 
            self.loss_Ds = self.loss_Ds / max(1,self.opt.n_acc_batches)
            self.loss_Ds.backward()
        if verbose:
            diagnose_network(self.netDs,"Ds")
        
        d_regularize = self.iter % self.reg_freq == 0

        if d_regularize:
            self.input.requires_grad = True
            real_pred = self.netDs(self.input)
            self.loss_r1 = d_r1_loss(real_pred, self.input)

            (.1 / 2 * self.loss_r1 * self.reg_freq).backward()


        # if self.augment and self.augment_p == 0:
        #     ada_augment_data = torch.tensor(
        #         (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
        #     )
        #     self.ada_augment += reduce_sum(ada_augment_data)

        #     if self.ada_augment[1] > 255:
        #         pred_signs, n_pred = self.ada_augment.tolist()

        #         r_t_stat = pred_signs / n_pred

        #         if r_t_stat > self.ada_target:
        #             sign = 1

        #         else:
        #             sign = -1

        #         self.ada_aug_p += sign * self.ada_aug_step * n_pred
        #         self.ada_aug_p = min(1, max(0, self.ada_aug_p))
        #         self.ada_augment.mul_(0)

        #update generator
        self.set_requires_grad(self.netDs, False)
        self.set_requires_grad(self.netG, True)
        update_G = self.backward_G_wgan(epoch)# if self.wgan else self.backward_G(epoch, train_threshold = self.opt.tlg)
        if update_G: 
            self.loss_G = self.loss_G / max(1,self.opt.n_acc_batches)
            self.loss_G.backward()            
        if verbose:
            diagnose_network(self.netG, "netG")

        # g_regularize = self.iter % self.reg_freq == 0
        # if g_regularize:
        #         self.path_batch_size = max(1, self.input.size(0))
        #         noise = mixing_noise(self.path_batch_size, self.latent, self.mixing, self.device)
        #         fake_img, latents = self.netG(noise, return_latents=True)

        #         path_loss, self.mean_path_length, path_lengths = g_path_regularize(
        #             fake_img, latents, self.mean_path_length
        #         )

        #         self.loss_path = .1 *self.reg_freq * path_loss

        #         self.loss_path.backward()

        #         self.mean_path_length_avg = (
        #             reduce_sum(self.mean_path_length).item() / get_world_size()
        #         )   


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

            return None