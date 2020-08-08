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
            layer += [nn.ReLU(0.2)]
            return layer

        encoder =[
            *conv_relu(3,ngf, stride = 1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            *conv_relu(ngf,ngf*2, stride = 1),
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
        #self.distance_kernel = kernels.LinearKernel()
        self.distance_kernel = kernels.ExponentialKernel(sigma = 1)

        decoder = [
            nn.Upsample(scale_factor=2),
            *conv_relu(npf, ngf*2, stride= 1),
            *conv_relu(ngf*2, ngf, stride= 1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, 3, kernel_size=3, bias=True, padding=1, stride=1, padding_mode="reflect"),
        ]
        self.decoder = nn.Sequential(*decoder)

        self.it = 0

    def positional_encoding(self,B,W,H, device = None):
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
        ref_pos = self.positional_encoding(B,W,H,device=x.device) #B,2,W,H
        pos = self.positional_encoding(B,W,H,device=x.device).view(B,2, -1)


        
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
                x_part = self.rule_mlp(x_part)
                #x_part, v = xv.split([C, 2], dim = 1)# B,C+2,WxH --> B,C,WxH; B,2,WxH

                v = self.velocity_mlp(x_part)
                #integrate position
                pos += v #.clamp(-1e5,1e5)


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
            pd = pos_e - ref_pos_e
           # print("p: " , pd.min().item(), pd.max().item(), (pd**2).min().item(), (pd**2).max().item())
           # print("d: ", dist.min().item(), dist.max().item(), "NAN" if (dist != dist).any() else "")
           # print("k: ", kdist.min().item(), kdist.max().item(), "NAN" if (kdist != kdist).any() else "")
            
            if i == 1 and self.debug: 
                for x in range (32): 
                    setattr(self.debug,f"heatmap_{x}", (dist[:,:,x,...] +1) /2 )
        
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

class LHCModel(BaseModel):
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

        self.num_heatmaps = 32
        for i in range(self.num_heatmaps):
            self.visual_names += [f"heatmap_{i}"]

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Gs', 'Gt', 'Ds', 'Dt', 'Ds_real','Ds_fake','Dt_real', 'Dt_fake']
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

  
        netG = LHC(latent_dim=32, ngf=32,npf = 64,steps_per_frame = 1, pd = 2, knn = 0, sample_kernel = "exp", nframes=self.nframes, debug = self)

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
            #self.optimizer_G = torch.optim.SGD(self.netG.parameters(), opt.lr)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_Ds = torch.optim.Adam(self.netDs.parameters(), lr=opt.lr * 1, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Ds)

            self.optimizer_Dt = torch.optim.Adam(self.netDt.parameters(), lr=opt.lr * 1, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Dt)


    def set_input(self, input):
        self.target_video = input['VIDEO'].to(self.device).permute(0,1,4,2,3).float() / 255.0 #normalize to [0,1] for vis and stuff
        if self.condition_gen:
            self.input = self.target_video[:,0,...]#first frame
        else:
            self.input = torch.empty((self.target_video.shape[0], self.in_dim)).normal_(mean=0, std=1)

        _, T, *_ = self.target_video.shape
        self.target_video = self.target_video[:, :min(T,self.nframes),...]

    def forward(self, frame_length = -1, train = True):
        if hasattr(self.netG, "nFrames"):
            self.netG.nFrames = frame_length if frame_length>0 else self.nframes
        if train and self.condition_gen:
            video = self.netG(self.target_video[:,:random.randrange(*self.train_range),...])
        else:
            video = self.netG(self.input)
        self.predicted_video = video
        #print(video.shape, self.target_video.shape)
        self.prediction_target_video = torch.cat([self.predicted_video[:, :self.nframes,...], self.target_video], dim = 4)

        for i in range(self.num_display_frames//2):
            setattr(self,f"frame_{i}", self.predicted_video[:,i,...] )
        for i in range(self.num_display_frames//2):
            ith_last = self.num_display_frames//2 -i +1
            setattr(self,f"frame_{i + self.num_display_frames//2}", self.predicted_video[:,-ith_last,...] )
        #video = video * 256
        #return video.permute(0,1,3,4,2)

    def epoch_frame_length(self, epoch):
        increase_intervals = 5
        iter_per_interval = 5
        return max(8,min(self.nframes // increase_intervals * (epoch // iter_per_interval +1), self.nframes))

    def gradient_penalty(self, real_img, fake_img, net):

        # Compute gradient penalty
        alpha = torch.rand(real_img.size(0), 1, 1, 1).cuda().expand_as(real_img)
        interpolated = torch.tensor(alpha * real_img.data + (1 - alpha) * fake_img.data, requires_grad=True)

        out = net(interpolated)

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        print(grad.size())
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        loss = self.lambda_gp * d_loss_gp
        return loss

    def sample_frames(self, vid, detach = False):
        _, T, *_ = self.predicted_video.shape
        frames=[]
        for i in random.sample(range(1, T), min(T-1,self.ndsframes)):
            f = vid[:,i,...]
            if detach:
                f = f.detach().to(self.device)
            if self.conditional:
                f = torch.cat([self.input, frame], dim = 1)
            frames.append(f)
        return torch.stack(frames, dim = 0)

    def backward_Ds_wgan(self, train_threshold = 0):
        self.loss_Ds_fake = 0
        self.loss_Ds_real = 0
        _, T, *_ = self.predicted_video.shape
        gp = 0
        n_samples = min(T-1,self.ndsframes)
        for i in random.sample(range(1, T), n_samples):

            fake,real = self.predicted_video[:,i,...].detach().to(self.device) , self.target_video[:,i,...]

            if self.conditional:
                fake = torch.cat([self.input, fake], dim = 1)
                real = torch.cat([self.input, real], dim = 1)
            # Fake
            # stop backprop to the generator by detaching fake_B
            pred_fake = self.netDs(fake)
            self.loss_Ds_fake += self.criterionWGAN(pred_fake, False)

            # Real
            pred_real = self.netDs(real)
            self.loss_Ds_real += self.criterionWGAN(pred_real, True)
       #     gp += self.gradient_penalty(real,fake,self.netDs)


        self.loss_Ds_real = self.loss_Ds_real / n_samples
        self.loss_Ds_fake = self.loss_Ds_fake / n_samples
        # Combined loss
        self.loss_Ds_GP = gp / n_samples
        self.loss_Ds = self.loss_Ds_real + self.loss_Ds_fake# + self.opt.lambda_GP * self.loss_Ds_GP
        self.loss_Ds.backward()
        return True

    def backward_Dt_wgan(self, train_threshold = 0):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = self.netDt(self.predicted_video.detach())
        self.loss_Dt_fake = self.criterionWGAN(pred_fake, False)
        # Real
        pred_real = self.netDt(self.target_video)
        self.loss_Dt_real = self.criterionWGAN(pred_real, True)

        # Combined loss
      #  self.loss_Dt_GP = self.gradient_penalty(self.target_video, self.predicted_video.detach(), net = self.netDt)

        self.loss_Dt = self.loss_Dt_real + self.loss_Dt_fake #+ self.opt.lambda_GP * self.loss_Dt_GP
        self.loss_Dt.backward()
        return True

    def backward_G_wgan(self, epoch, train_threshold = 0):
        # First, G(A) should fake the discriminator
        self.loss_G = 0
        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0

        for i in random.sample(range(1, self.nframes), self.ndsframes):
            fake = self.predicted_video[:,i,...]
            if self.conditional:
                fake = torch.cat([self.input, fake], dim = 1)

            pred_fake_ds = self.netDs(fake)
            self.loss_Gs += pred_fake_ds
        self.loss_Gs = torch.mean(self.loss_Gs / self.ndsframes)

        pred_fake_dt = self.netDt(self.predicted_video)
        self.loss_Gt = torch.mean(pred_fake_dt)

        self.loss_G += -(self.loss_Gs * self.opt.lambda_S + self.loss_Gt * self.opt.lambda_T)

        if epoch <= self.opt.pretrain_epochs:
            self.loss_G_L1 =self.criterionL1(self.predicted_video, self.target_video) * self.opt.lambda_L1
            self.loss_G += self.loss_G_L1

        self.loss_G.backward()
        return True

    def backward_Ds(self, train_threshold = 0):
        self.loss_Ds_fake = 0
        self.loss_Ds_real = 0
        self.loss_accDs_fake = 0
        self.loss_accDs_real = 0
        _, T, *_ = self.predicted_video.shape

        # Fake
        # stop backprop to the generator by detaching fake_B
        fake = self.sample_frames(self.predicted_video, detach=True)
        pred_fake = self.netDs(fake)
        self.loss_accDs_fake = torch.mean(1-pred_fake).item()
        self.loss_Ds_fake = self.criterionGAN(pred_fake, False)

        # Real
        real = self.sample_frames(self.target_video, detach=True)
        pred_real = self.netDs(real)
        self.loss_accDs_real = torch.mean(pred_real).item()
        self.loss_Ds_real = self.criterionGAN(pred_real, True)

        self.loss_acc_Ds = (self.loss_accDs_fake + self.loss_accDs_real)*.5
        self.loss_Ds_real = self.loss_Ds_real
        # Combined loss
        self.loss_Ds = (self.loss_Ds_fake + self.loss_Ds_real)*0.5

        if self.loss_acc_Ds < train_threshold:
            self.activity_diag[-1,self.activity_diag_indices["Ds"]] = self.activity_diag_colors["Ds"]
            self.loss_Ds.backward()
            return True
        return False

    def backward_Dt(self, train_threshold = 0):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = self.netDt(self.predicted_video.detach())
        self.loss_accDt_fake = torch.mean(1-pred_fake).item()
        self.loss_Dt_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netDt(self.target_video)
        self.loss_accDt_real = torch.mean(pred_real).item()
        self.loss_Dt_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_acc_Dt = (self.loss_accDt_fake + self.loss_accDt_real)*.5
        self.loss_Dt = (self.loss_Dt_fake + self.loss_Dt_real) * 0.5
        if self.loss_acc_Dt < train_threshold:
            self.activity_diag[-1,self.activity_diag_indices["Dt"]] = self.activity_diag_colors["Dt"]
            self.loss_Dt.backward()
            return True
        return False

    def backward_G(self, epoch, train_threshold = 0):
        # First, G(A) should fake the discriminator
        self.loss_G = 0
        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0

        samples = self.sample_frames(self.predicted_video, detach = False)
        pred_fake_ds = self.netDs(samples)
        self.loss_Gs = self.criterionGAN(pred_fake_ds, True)

        pred_fake_dt = self.netDt(self.predicted_video)
        self.loss_Gt = self.criterionGAN(pred_fake_dt, True)

        trust_Ds = 1 if self.loss_accDs_fake > train_threshold else 0
        trust_Dt = 1 if self.loss_accDt_fake > train_threshold else 0
        trust = trust_Ds + trust_Dt

        if trust_Ds > 0:
            self.activity_diag[-1,self.activity_diag_indices["Gs"]] = self.activity_diag_colors["Gs"]
        if trust_Dt > 0:
            self.activity_diag[-1,self.activity_diag_indices["Gt"]] = self.activity_diag_colors["Gt"]

        if trust>0:
            self.loss_G += (self.loss_Gs * self.opt.lambda_S * trust_Ds + self.loss_Gt * self.opt.lambda_T * trust_Dt) / (trust)

        if epoch <= self.opt.pretrain_epochs:
            self.loss_G_L1 =self.criterionL1(self.predicted_video, self.target_video) * self.opt.lambda_L1
            self.loss_G += self.loss_G_L1

        if trust > 0 or epoch <= self.opt.pretrain_epochs:
            self.loss_G.backward()
            return True

        return False

    def optimize_parameters(self, epoch_iter, verbose = False):
        self.iter += 1
        verbose = verbose or self.opt.verbose
        #Te = self.epoch_frame_length(epoch_iter)
        Te = self.nframes
        self.forward(frame_length=Te)
        _, T,*_ = self.predicted_video.shape
        _, TT,*_ = self.target_video.shape
        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0

        T = min(T,TT,Te) # just making sure to cut target if we didnt predict all the frames and to cut prediction, if we predicted more than target (i.e. we already messed up somewhere)

        #update Discriminator(s)
        self.set_requires_grad(self.netDs, True)
        self.optimizer_Ds.zero_grad()
        update_Ds = self.backward_Ds_wgan() if self.wgan else self.backward_Ds(train_threshold = self.opt.tld)
        if verbose:
            diagnose_network(self.netDs,"Ds")

        if update_Ds:
            nn.utils.clip_grad_norm_(self.netDs.parameters(), self.opt.clip_grads)
            self.optimizer_Ds.step()

        self.set_requires_grad(self.netDt, True)
        self.optimizer_Dt.zero_grad()
        update_Dt = self.backward_Dt_wgan() if self.wgan else self.backward_Dt(train_threshold = self.opt.tld)

        if verbose:
            diagnose_network(self.netDt,"Dt")
        if update_Dt:
            nn.utils.clip_grad_norm_(self.netDt.parameters(), self.opt.clip_grads)
            self.optimizer_Dt.step()

        # update Generator every n_critic steps
        if self.iter % self.opt.n_critic == 0:
            self.set_requires_grad(self.netDs, False)
            self.set_requires_grad(self.netDt, False)

            self.set_requires_grad(self.netG, True)
            self.optimizer_G.zero_grad()
            update_G = self.backward_G_wgan(epoch_iter, train_threshold = self.opt.tlg) if self.wgan else self.backward_G(epoch_iter, train_threshold = self.opt.tlg)
            if verbose:
                diagnose_network(self.netG, "netG")
            if update_G:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.clip_grads)
                self.optimizer_G.step()

        self.activity_diag_plt["X"] = list(range(self.activity_diag.shape[0]))
        self.activity_diag_plt["Y"] = self.activity_diag
        self.activity_diag = torch.cat([self.activity_diag, self.activity_diag[-1:]], dim = 0)

    def compute_losses(self, secs = 1, fps = 30):
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
            ## loss = L1(prediction - target)
            #print(torch.min(self.target_video), torch.max(self.target_video))
            loss_L1 = self.criterionL1(self.predicted_video, self.target_video) * self.opt.lambda_L1

            ds_perframe = {"opts": {
                    'title': "Loss per frame",
                    'legend': ["Ds prediction"],
                    'xlabel': 'frame',
                    'ylabel': 'loss',
                    } ,
          #          "Y":np.array((1,1)), "X":np.array((1,1))
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