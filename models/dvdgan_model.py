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


#import torch.nn.utils.spectral_norm as SpectralNorm

class DvdConditionalGenerator(nn.Module):

    def __init__(self, latent_dim=4, ch=32, enc_ch = 4, nframes=48, step_frames = 1, bn=True):
        super().__init__()
        self.step_frames = step_frames
        self.latent_dim = latent_dim
        self.ch = ch
        self.nframes = nframes -1 # first frame is just input frame
        self.n_steps = math.ceil(self.nframes / step_frames)

        self.encoder = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, enc_ch, kernel_size=(3, 3), padding=1)),
            GResBlock(enc_ch, enc_ch*4,n_class=1, downsample_factor = 2, bn = bn),
 #           GResBlock(2*enc_ch, 4*enc_ch, n_class=1, downsample_factor = 2, bn = bn),
            #GResBlock(2*enc_ch, 4*enc_ch, n_class=1, downsample_factor = 2, bn = bn),
            nn.AdaptiveAvgPool2d(latent_dim),
            SpectralNorm(nn.Conv2d(4*enc_ch, 8*ch, kernel_size=1)),
        )


        self.conv = nn.ModuleList([
            #ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            # GResBlock(8 * ch, 8 * ch, n_class=1, upsample_factor=1),
            # GResBlock(8 * ch, 8 * ch, n_class=1),
            #ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            # GResBlock(8 * ch, 8 * ch, n_class=1, upsample_factor=1, bn = bn),
            # GResBlock(8 * ch, 8 * ch, n_class=1, bn = bn),
            #ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch], kernel_sizes=3, n_layers=1),
            GResBlock(8 * ch, 8 * ch, n_class=1, upsample_factor=2, bn = bn),
            GResBlock(8 * ch, 4 * ch, n_class=1 ,bn = bn),
            #ConvGRU(4 * ch, hidden_sizes=[4 * ch, 8 * ch, 4 * ch], kernel_sizes=[3, 5, 5], n_layers=3),
            ConvGRU(4 * ch, hidden_sizes=[4 * ch], kernel_sizes=3, n_layers=1),
            GResBlock(4 * ch, 4 * ch, n_class=1, upsample_factor=1, bn = bn),
            GResBlock(4 * ch, 2 * ch, n_class=1, bn = bn)
        ])

        # TODO impl ScaledCrossReplicaBatchNorm
        # self.ScaledCrossReplicaBN = ScaledCrossReplicaBatchNorm2d(1 * chn)

        self.colorize = SpectralNorm(nn.Conv2d(2 * ch, 3, kernel_size=(3, 3), padding=1))
        #decode 1 RNN step into multiple frames using 3x3 convs

        if step_frames > 1:
            self.decoder = nn.Sequential(
                SpectralNorm(nn.Conv3d(2*ch, 2*ch, kernel_size=(3, 3, 3), padding=1)),
                nn.ReLU(),
                SpectralNorm(nn.Conv3d(2*ch, 2*ch, kernel_size=(3,3,3), padding=1)),
            )

    def forward(self, x):
        x = x * 2 - 1
        if len(x.shape) == 5: # B x T x 3 x W x H -> B x 3 x W x H (first frame)
            x = x[:,0,...]

        #y = y.view(-1, 8 * self.ch, self.latent_dim, self.latent_dim) # B x ch x ld x ld
        y = self.encoder(x)

        for k, conv in enumerate(self.conv):
            if isinstance(conv, ConvGRU):

                if k > 0:
                    _, C, W, H = y.size()
                    y = y.view(-1, self.n_steps, C, W, H).contiguous()

                frame_list = []
                for i in range(self.n_steps):
                    if k == 0:
                        if i == 0:
                            frame_list.append(conv(y))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y, frame_list[i - 1]))
                    else:
                        if i == 0:
                            frame_list.append(conv(y[:,0,:,:,:].squeeze(1)))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y[:,i,:,:,:].squeeze(1), frame_list[i - 1]))
                frame_hidden_list = []
                for i in frame_list:
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld

                y = y.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
                # print(y.size())
                B, T, C, W, H = y.size()
                y = y.view(-1, C, W, H)

            elif isinstance(conv, GResBlock):
                y = conv(y) # BT, C, W, H

        y = F.relu(y)
        BT, C, W, H = y.size()

        if self.step_frames > 1:
            y = y.view(-1, self.n_steps, C, W, H) # B, T/S, C, W, H
            y = y.permute(0, 2, 1, 3, 4) # B, C, T/S, W, H

            ysp = []
            for y_i in torch.split(y, split_size_or_sections = 1, dim = 2):
                y_i = F.interpolate(y_i, size = (self.step_frames,W,H))
                y_i = self.decoder(y_i)
                ysp.append(y_i)

            y = torch.cat(ysp, dim = 2)# B, C, T, W, H

        #    print(y.shape)
        #    y = F.interpolate(y, size = (self.step_frames,W,H))
        #     print(y.shape)
        #    y = self.decoder(y) # B, 3, T, H, W
            y = y.permute(0, 2, 1, 3, 4) [:,:self.nframes,...].contiguous() # B, T, C, W, H
            _,_, C, W, H = y.size()
            y = y.view(-1, C, W, H)


        y = self.colorize(y)
        y = y.view(-1, self.nframes,3, W, H) # B, T/S, C*S, W, H

        y = torch.tanh(y)

        y = torch.cat([x.unsqueeze(1), y],  dim = 1)
        y = (y+1) / 2.0 #[-1,1] -> [0,1] for vis
        return y



class DvdGanModel(BaseModel):
    def name(self):
        return 'DvdGanModel'

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
        self.num_display_frames = min(opt.num_display_frames, self.nframes - 1) //2 *2

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
        #netG = GRUEncoderDecoderNet(self.nframes,opt.input_nc,ngf = 32, hidden_dims=64, enc2hidden = True)
        #netG = GRUDeltaNet(self.nframes,opt.input_nc,hidden_dims=64)
        #default ch = 32
        self.in_dim = 1
        self.condition_gen = True
        self.conditional = False
        self.wgan = False
        if not self.wgan:
            self.loss_names += ['accDs_real','accDs_fake','accDt_real', 'accDt_fake']
        input_nc = opt.input_nc + (opt.num_segmentation_classes if opt.use_segmentation else 0)
      #  netG = DvdConditionalGenerator(nframes = self.nframes, ch = 32, enc_ch = 16, latent_dim = 8, step_frames = self.opt.unroll_frames, bn = bn)
        if opt.generator == "dvdgan":
            netG = DvdGenerator(nframes = self.nframes, ch = 8, n_class=1, in_dim = self.in_dim)
        elif opt.generator == "dvdgansimple":
            netG = GRUEncoderDecoderNet(self.nframes,input_nc ,ngf = 64, hidden_dims=32, enc2hidden = True)
        elif opt.generator == "trajgru": 
            netG = GRUEncoderDecoderNet(self.nframes,input_nc,ngf = 64, hidden_dims=32, enc2hidden = True, trajgru=True)

        else:
            assert False, f"unknown generator model specified: {opt.generator}!"
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.conditional = False
            #default chn = 128
            netDs = DvdSpatialDiscriminator(chn = 16, sigmoid = not self.wgan, cgan = self.conditional)
            self.netDs = networks.init_net(netDs, opt.init_type, opt.init_gain, self.gpu_ids)

            #default chn = 128
            netDt = DvdTemporalDiscriminator(chn = 24, sigmoid = not self.wgan)
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
        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0

    def set_input(self, input):
        self.target_video = input['VIDEO'].to(self.device).permute(0,1,4,2,3).float() / 255.0 #normalize to [0,1] for vis and stuff
        if self.condition_gen:
            self.input = self.target_video[:,0,...]#first frame
            if self.opt.use_segmentation: 
                self.input = torch.cat([self.input, input["SEGMENTATION"].to(self.device)])
        else:
            self.input = torch.empty((self.target_video.shape[0], self.in_dim)).normal_(mean=0, std=1)

        _, T, *_ = self.target_video.shape
        self.target_video = self.target_video[:, :min(T,self.nframes),...]

    def forward(self, frame_length = -1, train = True, sub_batch = -1):

        if hasattr(self.netG, "nFrames"):
            self.netG.nFrames = frame_length if frame_length>0 else self.nframes
        if sub_batch == -1: 
            lbi, ubi = 0, self.target_video.size(0)
        else: 
            lbi = sub_batch * self.parallell_batch_size #lower batch index
            ubi = min(lbi + self.parallell_batch_size, self.input.size(0))
        #if train and self.condition_gen:
        #    video = self.netG(self.target_video[lbi:ubi,:random.randrange(*self.train_range),...])
        #else:
        video = self.netG(self.input[lbi:ubi,...])
        self.predicted_video = video
        #print(video.shape, self.target_video.shape)
        vis = torch.cat([self.predicted_video[:, :self.nframes,...].detach().cpu(), self.target_video[lbi:ubi,...].detach().cpu()], dim = 4)
        
        if sub_batch <= 0:
            self.prediction_target_video = vis

            for i in range(self.num_display_frames//2):
                setattr(self,f"frame_{i}", self.predicted_video[:,i,...].detach().cpu() )
            for i in range(self.num_display_frames//2):
                ith_last = self.num_display_frames//2 -i +1
                setattr(self,f"frame_{i + self.num_display_frames//2}", self.predicted_video[:,-ith_last,...].detach().cpu() )
        else: 
            self.prediction_target_video = torch.cat([self.prediction_target_video, vis], dim = 0)

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
        self.loss_Ds.backward()

        return True

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
        self.loss_Dt.backward()
        return True

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

        self.loss_G += (self.loss_Gs * self.opt.lambda_S * trust_Ds + self.loss_Gt * self.opt.lambda_T * trust_Dt) / (trust)

        if epoch <= self.opt.pretrain_epochs:
            self.loss_G_L1 =self.criterionL1(self.predicted_video, self.target_video) * self.opt.lambda_L1
            self.loss_G += self.loss_G_L1

        self.loss_G.backward()
        return True


    def optimize_parameters(self, epoch_iter, verbose = False):
        self.iter += 1
        verbose = verbose or self.opt.verbose
        #Te = self.epoch_frame_length(epoch_iter)
        Te = self.nframes
        B, TT,*_ = self.target_video.shape

        acc_update_Ds = False
        acc_update_Dt = False
        acc_update_G = False

        for sub_batch in range(math.ceil(B/self.parallell_batch_size)): 
            self.forward(frame_length=Te, sub_batch=sub_batch)
            _, T,*_ = self.predicted_video.shape

            T = min(T,TT,Te) # just making sure to cut target if we didnt predict all the frames and to cut prediction, if we predicted more than target (i.e. we already messed up somewhere)

            # update Generator every n_critic steps
            if self.iter % self.opt.n_critic == 0 and self.iter > 50:
                self.set_requires_grad(self.netDs, False)
                self.set_requires_grad(self.netDt, False)

                self.set_requires_grad(self.netG, True)
                self.optimizer_G.zero_grad()
                update_G = self.backward_G_wgan(epoch_iter, train_threshold = self.opt.tlg) if self.wgan else self.backward_G(epoch_iter, train_threshold = self.opt.tlg)
                if verbose:
                    diagnose_network(self.netG, "netG")
                if update_G:
                    acc_update_G = True
                else: 
                    pass # clear graph? 
            else: 
                #update Discriminator(s)
                self.set_requires_grad(self.netDs, True)
                self.optimizer_Ds.zero_grad()
                update_Ds = self.backward_Ds_wgan() if self.wgan else self.backward_Ds(train_threshold = self.opt.tld)
                if verbose:
                    diagnose_network(self.netDs,"Ds")

                if update_Ds:
                    acc_update_Ds = True
                else:
                    pass #clear graph? 
        
                self.set_requires_grad(self.netDt, True)
                self.optimizer_Dt.zero_grad()
                update_Dt = self.backward_Dt_wgan() if self.wgan else self.backward_Dt(train_threshold = self.opt.tld)

                if verbose:
                    diagnose_network(self.netDt,"Dt")
                if update_Dt:
                    acc_update_Dt = True
                else: 
                    pass #clear graph? 

        if acc_update_Ds:
            nn.utils.clip_grad_norm_(self.netDs.parameters(), self.opt.clip_grads)
            self.optimizer_Ds.step()
            
        if acc_update_Dt: 
            nn.utils.clip_grad_norm_(self.netDt.parameters(), self.opt.clip_grads)
            self.optimizer_Dt.step()

        if acc_update_G: 
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.clip_grads)
            self.optimizer_G.step()

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