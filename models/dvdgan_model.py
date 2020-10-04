import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.container import ModuleList
from util.util import *
from .base_model import BaseModel
from . import networks
import numpy as np
import math
import functools
import random
from collections import OrderedDict, namedtuple
from dvdgan.Discriminators import SpatialDiscriminator as DvdSpatialDiscriminator
from dvdgan.Discriminators import TemporalDiscriminator as DvdTemporalDiscriminator
from dvdgan.Generator import Generator as DvdGenerator
from dvdgan.GResBlock import GResBlock, GResBlock3D
from dvdgan.ConvGRU import ConvGRU
from dvdgan.Normalization import SpectralNorm
from .dvdgansimple_model import GRUEncoderDecoderNet
import stylegan2.model as sg2model
from stylegan2.model import StyledConv

class DvdConditionalGenerator(nn.Module):
    def __init__(self, input_nc = 3, latent_dim=4, ch=8,n_grulayers=1,  nframes=48, step_frames = 1, bn=True, trajgru = False, norm = nn.BatchNorm2d, loss_ae = False, noise = False):
        super().__init__()
        self.step_frames = step_frames
        self.latent_dim = latent_dim
        self.ch = ch
        self.nframes = nframes -1 # first frame is just input frame
        self.n_steps = math.ceil(self.nframes / step_frames)
        self.loss_ae = loss_ae
        self.L_aux = 0
        self.noise = noise
        self.criterionAE = torch.nn.MSELoss()

        gru_kernels = [3,5,3][:n_grulayers]
        gru_hiddens = np.array([1,2,1])[:n_grulayers]
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                SpectralNorm(nn.Conv2d(input_nc, ch, kernel_size=(3, 3), padding=1)),
                GResBlock(ch, ch*2,n_class=1, downsample_factor = 2, bn = bn),
                ),
        #    GResBlock(2*ch, 4*ch, n_class=1, downsample_factor = 2, bn = bn),
        #    GResBlock(2*ch, 4*ch, n_class=1, downsample_factor = 2, bn = bn),
            GResBlock(ch*2, ch*4,n_class=1, downsample_factor = 2, bn = bn),
            GResBlock(ch*4, ch*8,n_class=1, downsample_factor = 2, bn = bn),
            GResBlock(ch*8, ch*8,n_class=1, downsample_factor = 2, bn = bn),
        #    SpectralNorm(nn.Conv2d(8*ch, 8*ch, kernel_size=1)),
        ])
        self.conv = nn.ModuleList([
            #ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            ConvGRU(8 * ch, hidden_sizes=8 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers),
            GResBlock(8 * ch, 8 * ch, n_class=1, upsample_factor=2, bn = bn, norm = norm),
            GResBlock(8 * ch, 8 * ch, n_class=1, upsample_factor=1, bn = bn, norm = norm),
            #ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            ConvGRU(8 * ch, hidden_sizes=8 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),
            GResBlock(8 * ch, 8 * ch, n_class=1, upsample_factor=2, bn = bn, norm = norm),
            GResBlock(8 * ch, 4 * ch, n_class=1, upsample_factor=1, bn = bn, norm = norm),
            #ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            ConvGRU(4 * ch, hidden_sizes=4 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),
            GResBlock(4 * ch, 4 * ch, n_class=1, upsample_factor=2, bn = bn, norm = norm),
            GResBlock(4 * ch, 2 * ch, n_class=1, upsample_factor=1, bn = bn, norm = norm),
            #ConvGRU(4 * ch, hidden_sizes=[4 * ch, 8 * ch, 4 * ch], kernel_sizes=[3, 5, 5], n_layers=3),
            ConvGRU(2 * ch, hidden_sizes=2 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),
            GResBlock(2 * ch, 2 * ch, n_class=1, upsample_factor=2, bn = bn, norm = norm),
            GResBlock(2 * ch, 1 * ch, n_class=1, upsample_factor=1, bn = bn, norm = norm)
        ])
        self.colorize = SpectralNorm(nn.Conv2d(1 * ch, 3, kernel_size=(3, 3), padding=1))
        #decode 1 RNN step into multiple frames using 3x3 convs
        if step_frames > 1:
            self.decoder = nn.Sequential(
                SpectralNorm(nn.Conv3d(1*ch, 1*ch, kernel_size=(3, 3, 3), padding=1)),
                nn.ReLU(),
                SpectralNorm(nn.Conv3d(1*ch, 1*ch, kernel_size=(3,3,3), padding=1)),
            )

    def forward(self, x, noise = None):
        x = x * 2 - 1
        if len(x.shape) == 5: # B x T x 3 x W x H -> B x 3 x W x H (first frame)
            x = x[:,0,...]

        encoder_list = [x]
        for layer in self.encoder: 
            encoder_list.append(layer(encoder_list[-1]))
        #y = self.encoder(x)
        encoder_list = encoder_list[1:]
        encoder_list.reverse()

        if self.noise: 
            if noise is None: 
                noise = torch.empty(encoder_list[0].shape, device=x.device).normal_(mean=0, std=1)
            y = noise.view(encoder_list[0].shape) # B x ch x ld x ld
        else: #use encoded frame
            y = encoder_list[0] # B x ch x ld x ld
        depth = 0
  
        for k, conv in enumerate(self.conv):
            if isinstance(conv, ConvGRU):
                if k > 0:
                    _, C, W, H = y.size()
                    y = y.view(-1, self.n_steps, C, W, H).contiguous()

                frame_list = []
                for i in range(self.n_steps):
                    if k == 0:
                        if i == 0:
                            frame_list.append(conv(y, [encoder_list[depth]]))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y, frame_list[i - 1]))
                    else:
                        if i == 0:
                            frame_list.append(conv(y[:,0,:,:,:].squeeze(1),[encoder_list[depth]]))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y[:,i,:,:,:].squeeze(1), frame_list[i - 1]))
                frame_hidden_list = []
                for i in frame_list:
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld

                y = y.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
                B, T, C, W, H = y.size()
                y = y.view(-1, C, W, H)
                depth += 1

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
            y = y.permute(0, 2, 1, 3, 4) [:,:self.nframes,...].contiguous() # B, T, C, W, H
            _,_, C, W, H = y.size()
            y = y.view(-1, C, W, H)

        frame_0 = x[:, :3, ...].unsqueeze(1)
        if self.loss_ae:
            frame_0 = 0
            up = 0
            for _, conv in enumerate(self.conv):
                if isinstance(conv, GResBlock):
                    if conv.upsample_factor == 2 and up < len(encoder_list): 
                        frame_0 += encoder_list[up]
                        up += 1   
                        #print(up, frame_0.shape, encoder_list[up].shape)
                    frame_0 = conv(frame_0)
                # elif isinstance(conv, ConvGRU):
                #     frame_0 = conv(frame_0)
            frame_0 = torch.tanh(self.colorize(frame_0))
            self.L_aux = self.criterionAE(frame_0, x[:, :3, ...])

            frame_0 = frame_0.unsqueeze(1)

        y = self.colorize(y)
        y = y.view(-1, self.nframes,3, W, H) # B, T/S, C*S, W, H
        y = torch.tanh(y)
        y = torch.cat([frame_0, y],  dim = 1)
        y = (y+1) / 2.0 #[-1,1] -> [0,1] for vis
        return y

class Dvd3DConditionalGenerator(nn.Module):
    def __init__(self, input_nc = 3, latent_dim=4, depth = 4, ch=8, nframes=48, bn=True, trajgru = False, norm = nn.BatchNorm3d, loss_ae = False, noise = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        self.nframes = nframes -1 # first frame is just input frame
        self.loss_ae = loss_ae
        self.L_aux = 0
        self.noise = noise
        self.criterionAE = torch.nn.MSELoss()
        self.depth = depth

        CH = [8,8,8,4,2,1]
        CH = CH[-depth-1:]

        self.encoder = nn.ModuleList([
            nn.Sequential(
                SpectralNorm(nn.Conv2d(input_nc, ch, kernel_size=(3, 3), padding=1)),
                GResBlock(ch*CH[-1], ch*CH[-2],n_class=1, downsample_factor = 2, bn = bn),
                ),
            *[GResBlock(ch*CH[-d-1], ch*CH[-d-2],n_class=1, downsample_factor = 2, bn = bn) for d in range(1,self.depth)]
        ])
        n_layers = 1
        rnn = []
        conv = []
        for d in range(self.depth): 
            c = CH[d]
            rnn.append(ConvGRU(c * ch, hidden_sizes=[c * ch], kernel_sizes=3, n_layers=n_layers),)
            conv.append(nn.Sequential(
                GResBlock3D(ch*c, ch*c, n_class=1, upsample_factor=2, bn = bn, norm = norm),
                GResBlock3D(ch*c, ch*CH[d+1], n_class=1, upsample_factor=1, bn = bn, norm = norm),
            ))
        self.rnn = nn.ModuleList(rnn)
        self.conv = nn.ModuleList(conv)

        self.colorize = SpectralNorm(nn.Conv2d(1 * ch, 3, kernel_size=(3, 3), padding=1))

    def forward(self, x, noise = None):
        x = x * 2 - 1
        if len(x.shape) == 5: # B x T x 3 x W x H -> B x 3 x W x H (first frame)
            x = x[:,0,...]
        encoder_list = [x]
        for layer in self.encoder: 
            encoder_list.append(layer(encoder_list[-1]))
        #y = self.encoder(x)
        encoder_list = encoder_list[1:]
        encoder_list.reverse()

        if self.noise: 
            if noise is None: 
                noise = torch.empty(encoder_list[0].shape, device=x.device).normal_(mean=0, std=1)
            y = noise.view(encoder_list[0].shape) # B x ch x ld x ld
        else: #use encoded frame
            y = encoder_list[0] # B x ch x ld x ld
        
        y = y.unsqueeze(2).expand(-1,-1, math.ceil(self.nframes / (2**(self.depth-1))), -1, -1)#B x ch x T//D x ld x ld

        for depth, (rnn, conv) in enumerate(zip(self.rnn, self.conv)): 
            unrolls = math.ceil(self.nframes / (2**(self.depth - depth)))
        #    print(f"----Depth: {depth} Unrolls: {unrolls} ----")
            frame_list = [encoder_list[depth]]
            for i in range(unrolls): 
                # print(y[:,:,i,:,:].shape, frame_list[i - 1].shape)
                frame_list.append(rnn(y[:,:,i,:,:], frame_list[i - 1]))
            
            y=torch.stack(frame_list, dim = 2)
            y = conv(y) #B x ch x T//D x ld x ld -> B x ch x T//(D-1) x ld*2 x ld*2 ->
      #      print(y.shape)

        y = y[:,:,:self.nframes,...]
        y = F.relu(y)
        B, C, T, W, H = y.size()
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, W, H)

        BT, C, W, H = y.size()
        frame_0 = x[:, :3, ...].unsqueeze(1)
        # if self.loss_ae:
        #     frame_0 = 0
        #     up = 0
        #     for _, conv in enumerate(self.conv):
        #         if isinstance(conv, GResBlock):
        #             if conv.upsample_factor == 2 and up < len(encoder_list): 
        #                 frame_0 += encoder_list[up]
        #                 up += 1   
        #                 #print(up, frame_0.shape, encoder_list[up].shape)
        #             frame_0 = conv(frame_0)
        #         # elif isinstance(conv, ConvGRU):
        #         #     frame_0 = conv(frame_0)
        #     frame_0 = torch.tanh(self.colorize(frame_0))
        #     self.L_aux = self.criterionAE(frame_0, x[:, :3, ...])

        #     frame_0 = frame_0.unsqueeze(1)

        y = self.colorize(y)
        y = y.view(-1, self.nframes,3, W, H) # B, T/S, C*S, W, H
        y = torch.tanh(y)
        y = torch.cat([frame_0, y],  dim = 1)
        y = (y+1) / 2.0 #[-1,1] -> [0,1] for vis
        return y

class DvdStyleConditionalGenerator(nn.Module):
    def __init__(self, input_nc = 3, latent_dim=2,n_grulayers = 1,style_dim = 256, ch=8, nframes=48, step_frames = 1, trajgru = False, noise = False):
        super().__init__()
        self.step_frames = step_frames
        self.latent_dim = latent_dim
        self.ch = ch
        self.nframes = nframes -1 # first frame is just input frame
        self.n_steps = math.ceil(self.nframes / step_frames)
        self.noise = noise
        self.criterionAE = torch.nn.MSELoss()
        bn = False
        self.n_grulayers = n_grulayers
        gru_kernels = [3,5,3][:n_grulayers]
        gru_hiddens = np.array([1,1,1])[:n_grulayers]

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ch, kernel_size=(3, 3), padding=1),
                GResBlock(ch, ch*2,n_class=1, downsample_factor = 2, bn = bn, weight_norm=None),
                ),
        #    GResBlock(2*ch, 4*ch, n_class=1, downsample_factor = 2, bn = bn),
            #GResBlock(2*ch, 4*ch, n_class=1, downsample_factor = 2, bn = bn),
            GResBlock(ch*2, ch*4,n_class=1, downsample_factor = 2, bn = bn, weight_norm=None),
            GResBlock(ch*4, ch*8,n_class=1, downsample_factor = 2, bn = bn, weight_norm=None),
            GResBlock(ch*8, ch*8,n_class=1, downsample_factor = 2, bn = bn, weight_norm=None),
        #    SpectralNorm(nn.Conv2d(8*ch, 8*ch, kernel_size=1)),
        ])
        e2s_dim = latent_dim // 4
        self.encoder2style = nn.Sequential(
                GResBlock(ch*8, ch*8,n_class=1, downsample_factor = 2, bn = bn, weight_norm=None),
                GResBlock(ch*8, ch*4,n_class=1, downsample_factor = 2, bn = bn, weight_norm=None),
                nn.Flatten(),
                nn.Linear(ch*4*e2s_dim*e2s_dim, style_dim),
                nn.Sigmoid(),
            )
        self.conv = nn.ModuleList([
            ConvGRU(8 * ch, hidden_sizes=8 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),
            StyledConv(8 * ch, 8 * ch, 3, style_dim, upsample=True),
            StyledConv(8 * ch, 8 * ch, 3, style_dim, upsample=False),
            ConvGRU(8 * ch, hidden_sizes=8 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),
            StyledConv(8 * ch, 8 * ch, 3, style_dim, upsample=True),
            StyledConv(8 * ch, 4 * ch, 3, style_dim, upsample=False),
            ConvGRU(4 * ch, hidden_sizes=4 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),
            StyledConv(4 * ch, 4 * ch, 3, style_dim, upsample=True),
            StyledConv(4 * ch, 2 * ch, 3, style_dim, upsample=False),
            ConvGRU(2 * ch, hidden_sizes=2 * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),
            StyledConv(2 * ch, 2 * ch, 3, style_dim, upsample=True),
            StyledConv(2 * ch, 1 * ch, 3, style_dim, upsample=False),
        ])
        self.colorize = nn.Conv2d(1 * ch, 3, kernel_size=(3, 3), padding=1)
        #decode 1 RNN step into multiple frames using 3x3 convs
        if step_frames > 1:
            self.decoder = nn.Sequential(
                nn.Conv3d(1*ch, 1*ch, kernel_size=(3, 3, 3), padding=1),
                nn.ReLU(),
                nn.Conv3d(1*ch, 1*ch, kernel_size=(3,3,3), padding=1),
            )

    def forward(self, x, noise = None):
        x = x * 2 - 1
        if len(x.shape) == 5: # B x T x 3 x W x H -> B x 3 x W x H (first frame)
            x = x[:,0,...]

        encoder_list = [x]
        for layer in self.encoder: 
            encoder_list.append(layer(encoder_list[-1]))
        #y = self.encoder(x)
        encoder_list = encoder_list[1:]
        encoder_list.reverse()

        if self.noise: 
            if noise is None: 
                noise = torch.empty(encoder_list[0].shape, device=x.device).normal_(mean=0, std=1)
            y = noise.view(encoder_list[0].shape) # B x ch x ld x ld
        else: #use encoded frame
            y = encoder_list[0] # B x ch x ld x ld
        
        style = self.encoder2style(y)
        style = style.unsqueeze(1).expand(-1, self.nframes, -1).contiguous().view(x.size(0)*self.nframes, -1) # BT x style

        depth = 0
  
        for k, conv in enumerate(self.conv):
            if isinstance(conv, ConvGRU):

                if k > 0:
                    _, C, W, H = y.size()
                    y = y.view(-1, self.n_steps, C, W, H).contiguous()

                frame_list = []
                for i in range(self.n_steps):
                    if k == 0:
                        if i == 0:
                            frame_list.append(conv(y, [encoder_list[depth]]*self.n_grulayers))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y, frame_list[i - 1]))
                    else:
                        if i == 0:
                            frame_list.append(conv(y[:,0,:,:,:].squeeze(1),[encoder_list[depth]]*self.n_grulayers))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y[:,i,:,:,:].squeeze(1), frame_list[i - 1]))
                frame_hidden_list = []
                for i in frame_list:
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld

                y = y.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
                B, T, C, W, H = y.size()
                y = y.view(-1, C, W, H)
                depth += 1

            elif isinstance(conv, StyledConv):
             #   print(y.shape, style.shape, conv)
                y = conv(y, style) # BT, C, W, H

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
            y = y.permute(0, 2, 1, 3, 4) [:,:self.nframes,...].contiguous() # B, T, C, W, H
            _,_, C, W, H = y.size()
            y = y.view(-1, C, W, H)

        frame_0 = x[:, :3, ...].unsqueeze(1)
        # if self.loss_ae:
        #     frame_0 = 0
        #     up = 0
        #     for _, conv in enumerate(self.conv):
        #         if isinstance(conv, GResBlock):
        #             if conv.upsample_factor == 2 and up < len(encoder_list): 
        #                 frame_0 += encoder_list[up]
        #                 up += 1   
        #                 #print(up, frame_0.shape, encoder_list[up].shape)
        #             frame_0 = conv(frame_0)
        #         # elif isinstance(conv, ConvGRU):
        #         #     frame_0 = conv(frame_0)
        #     frame_0 = torch.tanh(self.colorize(frame_0))
        #     self.L_aux = self.criterionAE(frame_0, x[:, :3, ...])

        #     frame_0 = frame_0.unsqueeze(1)

        y = self.colorize(y)
        y = y.view(-1, self.nframes,3, W, H) # B, T/S, C*S, W, H
        y = torch.tanh(y)
        y = torch.cat([frame_0, y],  dim = 1)
        y = (y+1) / 2.0 #[-1,1] -> [0,1] for vis
        return y

class DvdStyle2(nn.Module):
    def __init__(self, input_nc = 3, latent_dim=2,depth = 4,up_blocks_per_rnn = 1, n_grulayers = 1,style_dim = 256, ch=8, nframes=48, step_frames = 1, trajgru = False):
        super().__init__()
        self.step_frames = step_frames
        self.latent_dim = latent_dim
        self.ch = ch
        self.nframes = nframes -1 # first frame is just input frame
        self.n_steps = math.ceil(self.nframes / step_frames)
        self.criterionAE = torch.nn.MSELoss()
        self.depth = depth
        self.n_grulayers = n_grulayers
        self.skip_rnn = up_blocks_per_rnn

        gru_kernels = [3,5,3][:n_grulayers]
        gru_hiddens = np.array([1,1,1])[:n_grulayers]

        CH = [8,8,8,4,2,1]
        CH = CH[-depth-1:]

        self.encoder = nn.ModuleList([
            nn.Sequential(
                SpectralNorm(nn.Conv2d(input_nc, ch, kernel_size=(3, 3), padding=1)),
                GResBlock(ch*CH[-1], ch*CH[-2],n_class=1, downsample_factor = 2, bn = False, weight_norm=None),
                ),
            *[GResBlock(ch*CH[-d-1], ch*CH[-d-2],n_class=1, downsample_factor = 2, bn = False, weight_norm=None) for d in range(1,self.depth)]
        ])

        e2s_dim = latent_dim // 4
        self.encoder2style = nn.Sequential(
                GResBlock(ch*8, ch*8,n_class=1, downsample_factor = 2, bn = False, weight_norm=None),
                GResBlock(ch*8, ch*4,n_class=1, downsample_factor = 2, bn = False, weight_norm=None),
                nn.Flatten(),
                nn.Linear(ch*4*e2s_dim*e2s_dim, style_dim),
                nn.Sigmoid(),
            )
        self.input = sg2model.ConstantInput(ch*8,size=latent_dim)

        rnn = []
        conv_fp = []  
        for d in range(self.depth): 
            c = CH[d]
            if d % self.skip_rnn == 0: 
                rnn.append(ConvGRU(c * ch, hidden_sizes=c * ch * gru_hiddens, kernel_sizes=gru_kernels, n_layers=n_grulayers, trajgru=trajgru),)
            else: 
                rnn.append(None)
            conv = []
            conv.append(StyledConv(c * ch, c * ch, 3, style_dim, upsample=True))

            # for i in range(up_blocks_per_rnn -1): 
            #     conv.append(StyledConv(c * ch, c * ch * ch, 3, style_dim, upsample=False))
            #     conv.append(StyledConv(c * ch, c * ch, 3, style_dim, upsample=True))

            conv.append(StyledConv(c * ch, CH[d+1] * ch, 3, style_dim, upsample=False))

            conv_fp.append(nn.ModuleList(conv))
        self.rnn = nn.ModuleList(rnn)
        self.conv = nn.ModuleList(conv_fp)

        self.colorize = nn.Conv2d(1 * ch, 3, kernel_size=(3, 3), padding=1)
        #decode 1 RNN step into multiple frames using 3x3 convs
        if step_frames > 1:
            self.decoder = nn.Sequential(
                nn.Conv3d(1*ch, 1*ch, kernel_size=(3, 3, 3), padding=1),
                nn.ReLU(),
                nn.Conv3d(1*ch, 1*ch, kernel_size=(3,3,3), padding=1),
            )

    def forward(self, x, noise = None):
        x = x * 2 - 1
        if len(x.shape) == 5: # B x T x 3 x W x H -> B x 3 x W x H (first frame)
            x = x[:,0,...]

        encoder_list = [x]
        for layer in self.encoder: 
            encoder_list.append(layer(encoder_list[-1]))
        #y = self.encoder(x)
        encoder_list = encoder_list[1::self.skip_rnn]
        encoder_list.reverse()

        y = self.input(encoder_list[0]) #B x C x W x H
        style = self.encoder2style(y)
        style = style.unsqueeze(1).expand(-1, self.nframes, -1).contiguous().view(x.size(0)*self.nframes, -1) # BT x style
        y = y.unsqueeze(1).expand(-1, self.nframes, -1, -1, -1) #B x T x C x W x H

        for depth, (rnn, conv) in enumerate(zip(self.rnn, self.conv)): 
            if rnn: 
                frame_list = []
                frame_list = [encoder_list[depth]]
                if depth > 0:
                    _, C, W, H = y.size()
                    y = y.view(-1, self.nframes, C, W, H).contiguous()

                frame_list = [[encoder_list[depth]]*self.n_grulayers]
                for i in range(self.nframes):
                    print(depth, i, y.shape, [x.shape for x in frame_list[-1]])
                    frame_list.append(rnn(y[:,i,...].squeeze(1), frame_list[-1]))
                frame_hidden_list = []
                for i in frame_list[1:]: #collect hiddens of last rnn (ignore first frame =input)
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld
                y = y.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
                *_, C, W, H = y.size()
                y = y.view(-1, C, W, H)

            for layer in conv: 
                y = layer(y, style) # BT, C, W, H

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
            y = y.permute(0, 2, 1, 3, 4) [:,:self.nframes,...].contiguous() # B, T, C, W, H
            _,_, C, W, H = y.size()
            y = y.view(-1, C, W, H)

        frame_0 = x[:, :3, ...].unsqueeze(1)
        # if self.loss_ae:
        #     frame_0 = 0
        #     up = 0
        #     for _, conv in enumerate(self.conv):
        #         if isinstance(conv, GResBlock):
        #             if conv.upsample_factor == 2 and up < len(encoder_list): 
        #                 frame_0 += encoder_list[up]
        #                 up += 1   
        #                 #print(up, frame_0.shape, encoder_list[up].shape)
        #             frame_0 = conv(frame_0)
        #         # elif isinstance(conv, ConvGRU):
        #         #     frame_0 = conv(frame_0)
        #     frame_0 = torch.tanh(self.colorize(frame_0))
        #     self.L_aux = self.criterionAE(frame_0, x[:, :3, ...])

        #     frame_0 = frame_0.unsqueeze(1)

        y = self.colorize(y)
        y = y.view(-1, self.nframes,3, W, H) # B, T/S, C*S, W, H
        y = torch.tanh(y)
        y = torch.cat([frame_0, y],  dim = 1)
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
            parser.add_argument('--lambda_GP', type=float, default=1, help='weight for gradient penalty')
            parser.add_argument('--lambda_AUX', type=float, default=0, help='weight for aux loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.nframes = int(opt.max_clip_length * opt.fps / opt.skip_frames)
        self.num_display_frames = min(opt.num_display_frames, self.nframes - 1) //2 *2
        self.opt = opt
        self.iter = 1
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
        if self.opt.masked_update: 
            self.visual_names +=["mask"]
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Gs', 'Gt', 'Ds', 'Dt']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG', 'netDs', 'netDt']
            if opt.pretrain_epochs > 0:
                self.loss_names.append('G_L1')
        else:  # during test time, only load Gs
            self.model_names = ['netG']
        # load/define networks
        
        self.conditional = opt.conditional
        self.wgan = not opt.no_wgan
        if not self.wgan:
            self.loss_names += ['accDs_real','accDs_fake','accDt_real', 'accDt_fake']
        else: 
            self.loss_names += ['Ds_real', 'Ds_fake', 'Dt_real', 'Dt_fake', 'Ds_GP', 'Dt_GP']
        log_res = int(math.log(opt.resolution, 2))
        fp_depth = min(opt.max_fp, log_res - 3)
        latent_dim = 2**(log_res - fp_depth)
        input_nc = opt.input_nc + (opt.num_segmentation_classes if opt.use_segmentation else 0)
        if opt.generator == "dvdgan":
            netG = DvdConditionalGenerator(nframes = self.nframes,input_nc = input_nc,n_grulayers = opt.gru_layers,  ch = opt.ch_g, latent_dim = latent_dim, step_frames = 1, bn = not opt.no_bn, noise=not opt.no_noise, loss_ae=self.isTrain and self.opt.lambda_AUX>0)
            #netG = DvdConditionalGenerator(nframes = self.nframes,input_nc = input_nc, ch = 16, latent_dim = 8, step_frames = 1, bn = True, norm = nn.InstanceNorm2d,)

            #netG = DVDGan(self.nframes,input_nc ,ngf = 32, latent_nc=120,fp_levels = 3, res = self.opt.resolution, )
        elif opt.generator == "trajgru": 
            netG = DvdConditionalGenerator(nframes = self.nframes,input_nc = input_nc,n_grulayers = opt.gru_layers, ch = opt.ch_g, latent_dim = latent_dim, step_frames = 1, bn = not opt.no_bn, noise=not opt.no_noise, loss_ae=self.isTrain and self.opt.lambda_AUX>0, trajgru=True)
            # netG = DvdConditionalGenerator(nframes = self.nframes,input_nc = input_nc, ch = 16, latent_dim = 8, step_frames = 1, bn = True, norm = nn.InstanceNorm2d, trajgru=True)
        elif opt.generator == "style": 
            netG = DvdStyleConditionalGenerator(nframes = self.nframes,input_nc = input_nc,n_grulayers = opt.gru_layers, ch = opt.ch_g, latent_dim = latent_dim, step_frames = 1, noise = opt.style_noise)
        elif opt.generator == "styletraj": 
            netG = DvdStyleConditionalGenerator(nframes = self.nframes,input_nc = input_nc,n_grulayers = opt.gru_layers, ch = opt.ch_g, latent_dim = latent_dim, step_frames = 1, noise = opt.style_noise, trajgru=True)
        elif opt.generator == "dvdgansimple":
            netG = GRUEncoderDecoderNet(self.nframes,input_nc ,ngf = opt.ch_g, hidden_dims=128, enc2hidden = True)
        elif opt.generator == "dvdgan3d":
            netG = Dvd3DConditionalGenerator(nframes = self.nframes,input_nc = input_nc, ch = opt.ch_g, latent_dim = latent_dim, bn = not opt.no_bn, noise=not opt.no_noise, loss_ae=self.isTrain and self.opt.lambda_AUX>0)
        elif opt.generator == "style2": 
            netG = DvdStyle2(nframes = self.nframes, depth = fp_depth, up_blocks_per_rnn=opt.up_blocks_per_rnn, input_nc = input_nc,n_grulayers = opt.gru_layers, ch = opt.ch_g, latent_dim = latent_dim, step_frames = 1)
        elif opt.generator == "style2traj": 
            netG = DvdStyle2(nframes = self.nframes, depth = fp_depth, up_blocks_per_rnn=opt.up_blocks_per_rnn, input_nc = input_nc,n_grulayers = opt.gru_layers, ch = opt.ch_g, latent_dim = latent_dim, step_frames = 1, trajgru=True)
        else:
            assert False, f"unknown generator model specified: {opt.generator}!"
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain and self.opt.lambda_AUX > 0: # and hasattr(self.netG, "L_aux"): 
            self.loss_names += ["Gaux"]
            self.loss_Gaux = 0

        if self.isTrain:
            self.ndsframes = opt.dvd_spatial_frames
            assert self.nframes > self.ndsframes+1, "number of frames sampled for disc should be leq to number of total frames generated (length-1)"
       
            #default chn = 128
            netDs = DvdSpatialDiscriminator(chn = opt.ch_ds, sigmoid = not self.wgan, input_nc = (3 + input_nc) if self.conditional else 3 )
            self.netDs = networks.init_net(netDs, opt.init_type, opt.init_gain, self.gpu_ids)

            #default chn = 128
            netDt = DvdTemporalDiscriminator(chn = opt.ch_dt, sigmoid = not self.wgan, prepool=not opt.no_dt_prepool)
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
        if self.opt.masked_update: 
            self.mask = input["SEGMENTATION"].to(self.device).expand(-1, 3, -1, -1)
       # self.noise_input = torch.empty((self.target_video.shape[0], self.in_dim)).normal_(mean=0, std=1)

        _, T, *_ = self.target_video.shape
        self.target_video = self.target_video[:, :min(T,self.nframes),...]

    def forward(self, frame_length = -1, train = True):
        if hasattr(self.netG, "nFrames"):
            self.netG.nFrames = frame_length if frame_length>0 else self.nframes
 
        self.predicted_video = self.netG(self.input)
        if self.opt.masked_update: 
            B,T,C,W,H = self.predicted_video.size()
            self.predicted_video = torch.where(self.mask.byte().unsqueeze(1).expand(B,T,C,W,H),self.predicted_video, self.target_video[:,:1,...].expand(B,T,C,W,H))

        if self.target_video.size(1) >= self.nframes: 
            self.prediction_target_video = torch.cat([self.predicted_video[:, :self.nframes,...].detach().cpu(), self.target_video.detach().cpu()], dim = 4)
        else: 
            self.prediction_target_video = self.predicted_video[:, :self.nframes,...].detach().cpu()

        for i in range(self.num_display_frames//2):
            setattr(self,f"frame_{i}", self.predicted_video[:,i,...].detach().cpu() )
        for i in range(self.num_display_frames//2):
            ith_last = self.num_display_frames//2 -i +1
            setattr(self,f"frame_{i + self.num_display_frames//2}", self.predicted_video[:,-ith_last,...].detach().cpu() )
       
        # probs = self.input[:,3:,...].detach().cpu()
        # labelmap = torch.argmax(probs, dim=1, keepdim=True).expand(-1,3,-1,-1)
        # labels = torch.unique(labelmap)
        # print(f"labels: {labels}")

        # setattr(self,f"frame_{1}", labelmap.float()/182)

        #video = video * 256
        #return video.permute(0,1,3,4,2)

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

    def backward_Ds(self, train_threshold = 0):
        _, T, *_ = self.predicted_video.shape
        # Fake
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
        return True

    def backward_Dt(self, train_threshold = 0):
        # Fake
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
        return True

    def backward_G(self, epoch, train_threshold = 0):
        self.loss_G_L1 = 0
        samples = self.sample_frames(self.predicted_video, detach = False)
        pred_fake_ds = self.netDs(samples)
        self.loss_Gs = self.criterionGAN(pred_fake_ds, True)
        pred_fake_dt = self.netDt(self.predicted_video)
        self.loss_Gt = self.criterionGAN(pred_fake_dt, True)

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
            update_G = self.backward_G_wgan(epoch, train_threshold = self.opt.tlg) if self.wgan else self.backward_G(epoch, train_threshold = self.opt.tlg)
            if update_G: 
                self.loss_G = self.loss_G / self.opt.n_acc_batches
                self.loss_G.backward()            
            if verbose:
                diagnose_network(self.netG, "netG")
        else: 
            #update Discriminator(s)
            self.set_requires_grad(self.netDs, True)
            update_Ds = self.backward_Ds_wgan() if self.wgan else self.backward_Ds(train_threshold = self.opt.tld)
            if update_Ds: 
                self.loss_Ds = self.loss_Ds / self.opt.n_acc_batches
                self.loss_Ds.backward()
            if verbose:
                diagnose_network(self.netDs,"Ds")

            self.set_requires_grad(self.netDt, True)
            update_Dt = self.backward_Dt_wgan() if self.wgan else self.backward_Dt(train_threshold = self.opt.tld)
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