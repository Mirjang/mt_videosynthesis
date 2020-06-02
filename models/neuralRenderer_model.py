import torch
import torch.nn as nn
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import functools

################
###  HELPER  ###
################

INVALID_UV = -1.0


from torchvision import models
from collections import namedtuple

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = 0.5 * (X + 1.0) # map to [0,1]
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        #use_norm = False
        use_norm = not (norm_layer is None)

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
     
        if use_norm: downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        if use_norm: upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if use_norm: up = [uprelu, upconv, upnorm]
            else: up = [uprelu, upconv]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            if use_norm: down = [downrelu, downconv, downnorm]
            down = [downrelu, downconv]
            if use_norm: up = [uprelu, upconv, upnorm]
            else: up = [uprelu, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    #def forward(self, x, expressions):
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        #elif self.innermost:
        #    #todo incorporate expressions
        #    return torch.cat([x, self.model(x)], 1)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetRenderer(nn.Module):
    def __init__(self, renderer, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetRenderer, self).__init__()
        # construct unet structure
        if renderer=='UNET_8_level':
            num_downs = 8
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        elif renderer=='UNET_6_level':
            num_downs = 6
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        elif renderer=='UNET_5_level':
            num_downs = 5
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        else: #if renderer=='UNET_3_level':
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
            unet_block = UnetSkipConnectionBlock(ngf,     ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, feature_map):
        return self.model(feature_map)

class BlendRenderer(nn.Module): 
    def __init__(self, renderer, input_nc, NOUT, n_layers, back2front = True): 
        super(BlendRenderer, self).__init__()
        assert(NOUT < input_nc // n_layers)
        self.nFeatures = input_nc // n_layers
        self.nOUT = NOUT
        self.n_layers = n_layers
        self.dummy = nn.Conv2d(1,1,1) # so our optimizer has something to optimize
        self.back2front = back2front

    def forward(self, x): 
        # simple front to back blending, assumes tex_dims = 4
        
        if self.back2front:
            alpha = x[:, -1:, ...].clamp(0,1.0) #add some alpha bc. textures are initialized as 0 ->output would be 0 
            #last = (self.n_layers-1)*self.nFeatures
            output = x[:, -(self.nOUT+1):-self.nOUT,...] * alpha
            for d in reversed(range(self.n_layers-1)): 
                alpha_in = x[:, self.nFeatures*d+self.nOUT:self.nFeatures*d+self.nOUT+1, ...] .clamp(0,1.0)
                color = x[:, self.nFeatures*d:self.nFeatures*d+self.nOUT, ...]
                output = alpha_in * color + ((1-alpha_in)*alpha) *output
                alpha = alpha_in
            return output[:, 0:self.nOUT, ...]
        else: 

            alpha_in = x[:, self.nOUT:self.nOUT+1, ...].clamp(0,1.0) #add some alpha bc. textures are initialized as 0 ->output would be 0 
            #last = (self.n_layers-1)*self.nFeatures
            output = x[:, 0:self.nOUT,...] * alpha_in
            for d in range(self.n_layers-1): 
                alpha = x[:, self.nFeatures*d+self.nOUT:self.nFeatures*d+self.nOUT+1, ...] .clamp(.001,1.0)
                alpha = (1-alpha_in)*alpha
                output = output  + alpha * x[:, self.nFeatures*d:self.nFeatures*d+self.nOUT, ...] 
                alpha_in = alpha_in + alpha
            return output[:, 0:self.nOUT, ...]

class ResidualBlock(nn.Module): 
    def __init__(self, input_nc, inner_nc, output_nc, kernel_size = 1, norm_layer=nn.BatchNorm2d, is_first=False, is_last=False, activation = nn.ReLU(), dropout = 0.2):
        super(ResidualBlock, self).__init__()
        
        self.is_outer = is_first or is_last
        model = []
        model += [nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size, bias=True)]
        if(dropout>0): 
            model += [nn.Dropout(dropout)]

        model += [activation]
        model += [nn.Conv2d(inner_nc, output_nc, kernel_size=kernel_size, bias=True)]
        if(dropout>0): 
            model += [nn.Dropout(dropout)]
        if norm_layer:
            model += [norm_layer(output_nc)]

        self.model = nn.Sequential(*model)
        self.activation = nn.Tanh() if is_last else nn.ReLU() 

    def forward(self, x):
        if self.is_outer:
            return self.activation(self.model(x))
        else: 
            return self.activation(x + self.model(x))

class PerPixelRenderer(nn.Module): 
    def __init__(self, renderer, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_inout_convs = True):
        super(PerPixelRenderer, self).__init__()
        
        n_blocks = int(renderer.split("_")[1])
        
        norm_layer=None

        model = []
        #model += [ResidualBlock(input_nc, ngf, ngf, kernel_size=1, norm_layer=norm_layer, is_first=True)]
        if(use_inout_convs and input_nc != ngf): 
            model += [nn.Conv2d(input_nc, ngf, kernel_size=1, stride=1 , bias=True)]

        for _ in range(n_blocks): 
            model += [ResidualBlock(ngf, ngf, ngf, kernel_size=1, norm_layer=norm_layer)]

        #model += [ResidualBlock(ngf, ngf, output_nc, kernel_size=1, norm_layer=norm_layer, is_last=True)]
        if(use_inout_convs): 
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, stride=1, bias=True)]
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x): 
        return self.model(x)

class PerPixel2Renderer(nn.Module): 
    def __init__(self, renderer, input_nc, output_nc, dropout = 0., activation=nn.ReLU, ngf=64, norm_layer=nn.BatchNorm2d, use_inout_convs = True, use_in_convs = False, use_out_convs = False):
        super(PerPixel2Renderer, self).__init__()
        
        n_blocks = int(renderer.split("_")[1])
        
        norm_layer=None

        model = []
        #model += [ResidualBlock(input_nc, ngf, ngf, kernel_size=1, norm_layer=norm_layer, is_first=True)]
        if(use_inout_convs and input_nc != ngf) or use_in_convs: 
            model += [nn.Conv2d(input_nc, ngf, kernel_size=1, stride=1 , bias=True)]


        for _ in range(n_blocks): 
            model += [ResidualBlock(ngf, ngf, ngf, kernel_size=1, norm_layer=norm_layer)]
            ngf_next= max(ngf//2, 4)

            model += [nn.Conv2d(ngf, ngf_next, kernel_size=1, stride=1 , bias=True)]
            model += [nn.ReLU()]
            ngf = ngf_next
            self.output_nc = ngf
        
        #model += [ResidualBlock(ngf, ngf, output_nc, kernel_size=1, norm_layer=norm_layer, is_last=True)]
        if(use_inout_convs) or use_out_convs: 
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, stride=1, bias=True)]
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x): 
        return self.model(x)

class PerPixel2bRenderer(nn.Module): 
    def __init__(self, renderer, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_inout_convs = True, use_in_convs = False, use_out_convs = False):
        super(PerPixel2bRenderer, self).__init__()
        
        n_blocks = int(renderer.split("_")[1])
        print("inc:"+str(input_nc))
        return
        norm_layer=None

        model = []
        #model += [ResidualBlock(input_nc, ngf, ngf, kernel_size=1, norm_layer=norm_layer, is_first=True)]
        if(use_inout_convs and input_nc != ngf) or use_in_convs: 
            model += [nn.Conv2d(input_nc, ngf, kernel_size=1, stride=1 , bias=True)]


        for _ in range(n_blocks): 
            model += [ResidualBlock(ngf, ngf, ngf, activation = nn.LeakyReLU(0.2), dropout = 0, kernel_size=1, norm_layer=norm_layer)]
            ngf_next= max(ngf//2, 16)

            model += [nn.Conv2d(ngf, ngf_next, kernel_size=1, stride=1 , bias=True)]
            model += [nn.LeakyReLU(0.2)]
            ngf = ngf_next
            self.output_nc = ngf
        
        #model += [ResidualBlock(ngf, ngf, output_nc, kernel_size=1, norm_layer=norm_layer, is_last=True)]
        if(use_inout_convs) or use_out_convs: 
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, stride=1, bias=True)]
            model += [nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, x): 
        return self.model(x)

class RnnPerPixelRenderer(nn.Module): 
    def __init__(self, renderer, output_nc, opt, norm_layer=nn.BatchNorm2d):
        super(RnnPerPixelRenderer, self).__init__()
        self.n_layers_per_iter = opt.extrinsics_skip #number of texture layers to be input at once into the RNN
        assert opt.num_depth_layers%self.n_layers_per_iter == 0, "Expecting even nr of depth layers and scene to consist of proper 3d objects (no planes)"
        self.n_layers = opt.num_depth_layers
        self.n_extrinsics = 1 if opt.use_spherical_harmonics else 3 if opt.use_extrinsics else 0
        self.tex_channels = opt.tex_features
        ngf = opt.nrhf
        nref = opt.nref
        nrdf = opt.nrdf
        self.isLstm = renderer.startswith("Lstm")

        ed_blocks = renderer.split("_")
        n_encoder_blocks = int(ed_blocks[1])
        n_decoder_blocks = int(ed_blocks[2])

        encoder = []
        encoder += [nn.Conv2d(self.n_extrinsics + opt.tex_features, nref, kernel_size=1, stride=1, bias=True)]
        for i in range(n_encoder_blocks): 
            encoder += [ResidualBlock(nref, nref, nref, kernel_size=1, norm_layer=norm_layer)]

        self.encoder = nn.Sequential(*encoder)
        
        if(self.isLstm):
            self.recurrent_cell = networks.ConvLSTMCell((opt.fineSize, opt.fineSize), nref, ngf, (1, 1), True)
        else: 
            self.recurrent_cell = networks.ConvGRUCell((opt.fineSize, opt.fineSize), nref, ngf, (1, 1), True)

        self.hidden_dims = (opt.batch_size, ngf, opt.fineSize,opt.fineSize)
        decoder = []

        # for i in range(n_decoder_blocks): 
        #     decoder += [ResidualBlock(ngf, ngf, ngf, kernel_size=1, norm_layer=norm_layer)]
        #     ngf = max(16, ngf//2)
        decoder += [PerPixel2Renderer("PerPixel_"+str(n_decoder_blocks), ngf, output_nc, ngf = nrdf, norm_layer=norm_layer)]
 
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x):
  
        if self.n_extrinsics>0: 
            extrinsics = x[:, :self.n_extrinsics,...]
        x.device
        h = torch.zeros(self.hidden_dims).to(x.device)
        if self.isLstm:
            c = torch.zeros(self.hidden_dims).to(x.device)

        for i in reversed(range(self.n_layers)): 
            l = self.n_extrinsics + i * self.tex_channels
            u = l + self.tex_channels
            x_i = x[:,l:u,...]
            if self.n_extrinsics>0: 
                x_i = torch.cat([extrinsics, x_i], 1)

            x_i = self.encoder(x_i)
            if self.isLstm:
                h, c = self.recurrent_cell(x_i, (h,c))
            else: 
                h = self.recurrent_cell(x_i,h)

        return self.decoder(h)

class BlendPerPixelRenderer(nn.Module): 
    def __init__(self, renderer, output_nc, opt, norm_layer=nn.BatchNorm2d):
        super(BlendPerPixelRenderer, self).__init__()
        self.n_layers_per_iter = opt.extrinsics_skip #number of texture layers to be input at once into the RNN
        assert opt.num_depth_layers%self.n_layers_per_iter == 0, "Expecting even nr of depth layers and scene to consist of proper 3d objects (no planes)"
        self.n_layers = opt.num_depth_layers // self.n_layers_per_iter
        self.n_extrinsics = 1 if opt.use_spherical_harmonics else 3 if opt.use_extrinsics else 0
        self.tex_channels = opt.tex_features * opt.extrinsics_skip + self.n_extrinsics
        self.back2front = False

        nref = opt.nref
        nrdf = opt.nrdf
        ed_blocks = renderer.split("_")
        n_encoder_blocks = int(ed_blocks[1])
        n_decoder_blocks = int(ed_blocks[2])

        encoder = [PerPixel2Renderer("PerPixel2_"+str(n_encoder_blocks), self.tex_channels, -1, ngf = nref, norm_layer=norm_layer, use_inout_convs=False, use_in_convs=True)]
        self.encoder = nn.Sequential(*encoder)
        self.blend_dims = (opt.batch_size, encoder[0].output_nc-1, opt.fineSize,opt.fineSize)
        self.alpha_dims = (opt.batch_size, 1, opt.fineSize,opt.fineSize)

        decoder = [PerPixel2Renderer("PerPixel2_"+str(n_decoder_blocks), encoder[0].output_nc-1, output_nc, ngf = nrdf, norm_layer=norm_layer, use_inout_convs=True)]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        h = torch.zeros(self.blend_dims).to(x.device)
        if self.back2front: 
            alpha = torch.ones(self.alpha_dims).to(x.device)
            for i in reversed(range(self.n_layers)): 
                l = i * self.tex_channels
                x_i = x[:,l:l + self.tex_channels,...]           
                x_i = self.encoder(x_i)
                alpha_in = torch.sigmoid(x_i[:,:1,...])
                x_i = x_i[:,1:,...]
                h = alpha_in * x_i + ((1-alpha_in)*alpha) * h
                alpha = alpha_in
        else: 
            alpha_in = torch.zeros(self.alpha_dims).to(x.device)
            for i in range(self.n_layers): 
                l = i * self.tex_channels
                x_i = x[:,l:l + self.tex_channels,...]           
                x_i = self.encoder(x_i)
                alpha = torch.sigmoid(x_i[:,:1,...])
                x_i = x_i[:,1:,...]
                d_alpha = (1-alpha_in)*alpha
                h = alpha_in * h + d_alpha * x_i
                alpha_in = alpha_in + d_alpha
        return self.decoder(h)

class RnnUNETRenderer(nn.Module): 
    def __init__(self, renderer, output_nc, opt, norm_layer=nn.BatchNorm2d, use_dropout = False):
        super(RnnUNETRenderer, self).__init__()
        self.n_layers = opt.num_depth_layers
        self.n_extrinsics = 1 if opt.use_spherical_harmonics else 3 if opt.use_extrinsics else 0
        self.tex_channels = opt.tex_features
        ngf = opt.ngf
        

        ed_blocks = renderer.split("_")
        n_encoder_blocks = int(ed_blocks[1])
        n_decoder_layers = int(ed_blocks[2])

        encoder = []
        encoder += [nn.Conv2d(self.n_extrinsics + opt.tex_features, ngf, kernel_size=1, stride=1, bias=True)]
        for i in range(n_encoder_blocks): 
            encoder += [ResidualBlock(ngf, ngf, ngf, kernel_size=1, norm_layer=norm_layer)]

        self.encoder = nn.Sequential(*encoder)

        if(renderer.startswith("Lstm")):
            self.recurrent_cell = networks.ConvLSTMCell((opt.fineSize, opt.fineSize), ngf, ngf, (1, 1), True)
        else: 
            self.recurrent_cell = networks.ConvGRUCell((opt.fineSize, opt.fineSize), ngf, ngf, (1, 1), True)
        
        self.hidden_dims = (opt.batch_size, ngf, opt.fineSize,opt.fineSize)

        #def __init__(self, renderer, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        self.decoder = UnetRenderer("UNET_"+str(n_decoder_layers)+"_level", ngf, output_nc, opt.ngf, norm_layer=norm_layer, use_dropout=use_dropout)


    def forward(self, x):
  
        if(self.n_extrinsics>0): 
            extrinsics = x[:, :self.n_extrinsics,...]
        x.device
        h = torch.zeros(self.hidden_dims).to(x.device)
        c = torch.zeros(self.hidden_dims).to(x.device)

        for i in reversed(range(self.n_layers)): 
            l = self.n_extrinsics + i * self.tex_channels
            u = l + self.tex_channels
            x_i = x[:,l:u,...]
            if(self.n_extrinsics>0): 
                x_i = torch.cat([extrinsics, x_i], 1)

            x_i = self.encoder(x_i)
            h, c = self.lstm(x_i, (h,c))

        return self.decoder(h)


def define_Renderer(renderer, n_feature, opt, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    ngf = opt.ngf
    
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm)
    N_OUT = 3
    #net = UnetRenderer(N_FEATURE, N_OUT, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    if(renderer.startswith("UNET")):
        net = UnetRenderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif(renderer.startswith("PerPixel2b")):
        net = PerPixel2Renderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer)
    elif(renderer.startswith("PerPixel2")):
        net = PerPixel2Renderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer)
    elif(renderer.startswith("PerPixel")):
        net = PerPixelRenderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer)
    elif(renderer.startswith("BlendPerPixel")):
        net = BlendPerPixelRenderer(renderer, N_OUT, opt)
    elif(renderer.startswith("Blend")):
        net = BlendRenderer(renderer, n_feature, N_OUT, opt.num_depth_layers)
    elif(renderer.startswith("LstmPerPixel") or renderer.startswith("GruPerPixel")):
        net = RnnPerPixelRenderer(renderer, N_OUT, opt)
    elif(renderer.startswith("LstmUNET") or renderer.startswith("GruUNET")):
        net = RnnUNETRenderer(renderer, N_OUT, opt, use_dropout=use_dropout)
    return networks.init_net(net, init_type, init_gain, gpu_ids)

class Texture(nn.Module):
    def __init__(self, n_textures, n_features, dimensions, device, id_mapping):
        super(Texture, self).__init__()
        self.device = device
        self.n_textures = n_textures
        #self.register_parameter('data', torch.nn.Parameter(torch.randn(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True)))
        #self.register_parameter('data', torch.nn.Parameter(2.0 * torch.ones(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True) -1.0))
        #self.register_parameter('data', torch.nn.Parameter(torch.zeros(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True)))
        self.register_parameter('data', torch.nn.Parameter(2.0 * torch.ones(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True) -1.5))
        self.id_mapping = id_mapping

    def unfuck(self, world_positions):
                        
        for texture_id in range(9): 
            self.data[texture_id, :3] = world_positions[texture_id]

    def forward(self, uv_inputs, mask_inputs, world_positions, extrinsics, extrinsics_type=None, extrinsics_skip = 1):

        layers = []
        N, n_layers, H, W =mask_inputs.shape
        _, F, *_ = self.data.shape
        #print(extrinsics_type)
        if extrinsics_type: 
            F += 3

        for layer in range(n_layers): 
            layer_idx = 2*layer
            mask_layer = mask_inputs[:,layer,:,:]
            uvs = torch.stack([uv_inputs[:,layer_idx,:,:], uv_inputs[:,layer_idx+1,:,:]], 3)
            uvs_wp = torch.stack([1-uv_inputs[:,layer_idx,:,:], 1-uv_inputs[:,layer_idx+1,:,:]], 3)

            layer_tex = torch.zeros((N,F,H,W), device=self.device)
            objects_in_mask = torch.unique(mask_layer).detach()
            #for texture_id in range(self.n_textures): 
            for texture_id in objects_in_mask: 
                mask = mask_layer == texture_id                    
                if self.id_mapping: 
                    texture_id = torch.tensor(self.id_mapping[texture_id]).to(self.device)
                #background is 0 in mask and has no texture atm
                if texture_id < 1 : # just keep background (void) as zero
                    if texture_id < 0:
                        print("Invalid tex_id!")
                    continue
                sample = torch.nn.functional.grid_sample(self.data[texture_id:texture_id+1, :, :, :], uvs, mode='bilinear', padding_mode='border') #, align_corners = False)

                if extrinsics_type and layer%extrinsics_skip==0: 
                    wp_sample = torch.nn.functional.grid_sample(world_positions[texture_id:texture_id+1, :, :, :], uvs, mode='bilinear', padding_mode='border') #, align_corners = False)
                    wp_sample = wp_sample - extrinsics
                    norm = torch.norm(wp_sample, p = 2, dim = 1).detach()
                    view_dir = wp_sample.div(norm.expand_as(wp_sample))
                    sample = torch.cat([sample, view_dir ], 1)   

                layer_tex = layer_tex + sample * mask.float()
            if extrinsics_type == "SH" and layer%extrinsics_skip==0: 
                assert(F>11) # we need 8 channels for SH + 3 extrinsics channels + at least 1 texture channel 
                layer_extrinsics = layer_tex[:, -3:, ...]
                layer_tex = layer_tex[:, :-3,...]

                layer_tex = self.sh_Layer(layer_tex, layer_extrinsics)
            layers.append(layer_tex)
        return torch.cat(layers, 1)

    def sh_Layer(self, tex, extrinsics):
        dir = extrinsics[0]
        dir_x = dir[0]
        dir_y = dir[1]
        dir_z = dir[2]

        sh_band_0   = float(1.0)
        sh_band_1_0 = dir_y
        sh_band_1_1 = dir_z
        sh_band_1_2 = dir_x
        sh_band_2_0 = dir_x * dir_y
        sh_band_2_1 = dir_y * dir_z
        sh_band_2_2 = (3.0 * dir_z * dir_z - 1.0)
        sh_band_2_3 = dir_x * dir_z
        sh_band_2_4 = (dir_x * dir_x - dir_y * dir_y)

        return torch.cat([  tex[:,0:3,:,:],
                            sh_band_0 * tex[:,3+0:3+1,:,:], 
                            sh_band_1_0 * tex[:,3+1:3+2,:,:], sh_band_1_1 * tex[:,3+2:3+3,:,:], sh_band_1_2 * tex[:,3+3:3+4,:,:], 
                            sh_band_2_0 * tex[:,3+4:3+5,:,:], sh_band_2_1 * tex[:,3+5:3+6,:,:], sh_band_2_2 * tex[:,3+6:3+7,:,:], sh_band_2_3 * tex[:,3+7:3+8,:,:], sh_band_2_4 * tex[:,3+8:3+9,:,:],
                            tex[:,12:,:,:]
                            ], 1)

class HierarchicalTexture(nn.Module):
    def __init__(self, n_textures, n_features, dimensions, device):
        super(HierarchicalTexture, self).__init__()
        self.dim = dimensions
        self.register_parameter('data', torch.nn.Parameter(torch.zeros(n_textures, n_features, 2 * dimensions, dimensions, device=device, requires_grad=True)))

    def forward(self, uv_inputs, texture_id):
        uvs = torch.stack([uv_inputs[:,0,:,:], uv_inputs[:,1,:,:]], 3)

        # hard coded pyramid
        offsetY = 0
        w = self.dim
        self.high_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        high_level = torch.nn.functional.grid_sample(self.high_level_tex, uvs, mode='bilinear', padding_mode='border')
        offsetY += w
        w = w // 2
        self.medium_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        medium_level = torch.nn.functional.grid_sample(self.medium_level_tex, uvs, mode='bilinear', padding_mode='border')
        offsetY += w
        w = w // 2
        self.low_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        low_level = torch.nn.functional.grid_sample(self.low_level_tex, uvs, mode='bilinear', padding_mode='border')
        offsetY += w
        w = w // 2
        self.lowest_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        lowest_level = torch.nn.functional.grid_sample(self.lowest_level_tex, uvs, mode='bilinear', padding_mode='border')

        return high_level + medium_level + low_level + lowest_level

def define_Texture(n_textures, n_features, dimensions, device, gpu_ids=[], id_mapping = None):
    tex = Texture(n_textures, n_features, dimensions, device, id_mapping)

    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     tex.to(gpu_ids[0])
    #     tex = torch.nn.DataParallel(tex, gpu_ids)
    return tex

def define_HierarchicalTexture(n_textures, n_features, dimensions, device, gpu_ids=[]):
    tex = HierarchicalTexture(n_textures, n_features, dimensions, device)

    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     tex.to(gpu_ids[0])
    #     tex = torch.nn.DataParallel(tex, gpu_ids)
    return tex


def spherical_harmonics_basis(dir):
    # assumes that dir is a normalized direction vector with 3 components
    # output: vector with 9 coeffs
    dir_x = dir[0]
    dir_y = dir[1]
    dir_z = dir[2]

    sh_band_0   = float(1.0)
    sh_band_1_0 = dir_y
    sh_band_1_1 = dir_z
    sh_band_1_2 = dir_x
    sh_band_2_0 = dir_x * dir_y
    sh_band_2_1 = dir_y * dir_z
    sh_band_2_2 = (3.0 * dir_z * dir_z - 1.0)
    sh_band_2_3 = dir_x * dir_z
    sh_band_2_4 = (dir_x * dir_x - dir_y * dir_y)
    return np.array([sh_band_0,  sh_band_1_0, sh_band_1_1, sh_band_1_2,  sh_band_2_0, sh_band_2_1, sh_band_2_2, sh_band_2_3, sh_band_2_4], dtype=np.float32)

class NeuralRendererModel(BaseModel):
    def name(self):
        return 'NeuralRendererModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        #parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(norm='instance', netG='unet_256')
        #parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_VGG', type=float, default=100.0, help='weight for VGG loss')
        parser.add_argument('--lambda_GAN', type=float, default=100.0, help='weight for GAN loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.trainRenderer = opt.isTrain
        self.n_layers = opt.num_depth_layers
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        #self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.use_gan = self.opt.lossType == 'GAN' # or self.opt.lossType == 'all'

        self.loss_names = []
        if opt.lossType == 'L1':
            self.loss_names += ['G_L1']
        elif opt.lossType == 'VGG':
            self.loss_names += ['G_VGG']    
        elif opt.lossType == 'GAN':
            self.loss_names += ['G_GAN', 'G_L1', 'G_total', 'D_real', 'D_fake']
        elif opt.lossType == 'all':     
            self.loss_names += ['G_L1','G_VGG', 'G_total']

        self.loss_names += ['dummy']
        self.loss_dummy = 0
        self.world_positions = None
        self.visual_names = []
        self.nObjects = opt.nObjects
        self.use_extrinsics = opt.use_extrinsics
        if(opt.isTrain):
            for i in range(1,self.nObjects):
                self.visual_names.append(str("texture"+str(i)+"_col"))
            self.visual_names += ['sampled_texture_col']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names += ['fake', 'target']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG', 'texture']
            if self.use_gan: 
                self.model_names +=['netD']
        else:  # during test time, only load Gs
            self.model_names = ['netG', 'texture']

        ntf = opt.tex_features
        # load/define networks
        self.input_channels = opt.tex_features * opt.num_depth_layers 
        if opt.use_spherical_harmonics: 
            self.input_channels += 9*opt.num_depth_layers 
            ntf += 9
        elif opt.use_extrinsics:
            self.input_channels += 3*opt.num_depth_layers

        self.netG = define_Renderer(opt.rendererType, self.input_channels, opt, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #self.netG = define_Renderer(opt.rendererType, opt.tex_features+2, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)#<<<<<<<<<<<<<<<<
        # texture
        self.texture = define_Texture(self.nObjects, ntf, opt.tex_dim, device=self.device, gpu_ids=self.gpu_ids, id_mapping = opt.id_mapping)

        if self.isTrain:
            use_sigmoid = True
            if self.use_gan:
                # disc input: uv maps + masks + generator output
                self.netD = networks.define_D(3 * opt.num_depth_layers + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                #self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

                self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            if self.opt.lossType == 'VGG' or self.opt.lossType == 'all':
                self.vgg = VGG16().to(self.device)
            # initialize optimizers
            self.optimizers = []
            if self.trainRenderer:
            
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
                self.optimizers.append(self.optimizer_G)
                if self.use_gan:
                    self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_D)

            self.optimizer_T = torch.optim.Adam(self.texture.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_T)


            # print('type netG:')
            # for param in self.netG.parameters():
            #     print(type(param.data), param.size())
            # print('type texture:')
            # for param in self.texture.parameters():
            #     print(type(param.data), param.size())
            # print('-----')


    def set_input(self, input):
        self.target = input['TARGET'].to(self.device)
        self.input_uv = input['UV'].to(self.device)
        self.input_mask = input['MASK'].to(self.device)
        self.image_paths = input['paths']
        self.extrinsics = input['extrinsics']
        if self.world_positions is None or self.opt.update_world_pos:
            self.world_positions = input['worldpos'][0].to(self.device)
        if self.use_gan: 
            self.input_d = torch.cat((self.input_uv, self.input_mask.float() / self.nObjects), dim = 1)


    def forward(self):
        _,_, H, W = self.input_mask.shape
        if self.opt.use_spherical_harmonics:
            extrinsics_layer = self.extrinsics.unsqueeze(2).unsqueeze(3).expand(-1,-1, H,W).to(self.device)
            self.sampled_texture = self.texture(self.input_uv, self.input_mask, self.world_positions, extrinsics_layer, extrinsics_type="SH")
        elif self.opt.use_extrinsics: 
            extrinsics_layer = self.extrinsics.unsqueeze(2).unsqueeze(3).expand(-1,-1, H,W).to(self.device)
            self.sampled_texture = self.texture(self.input_uv, self.input_mask, self.world_positions, extrinsics_layer, extrinsics_type="DIR")
        else: 
            self.sampled_texture = self.texture(self.input_uv, self.input_mask, None, None)

        #first layer first 3 channels, rgb channels for nth layers will be [:, nFeatures*n:nFeatures*n+1, ...]
        self.sampled_texture_col = self.sampled_texture[:,:3,:,:]

        #set textures for visualizer. texture0 = background (no uv map)
        for i in range(1,self.nObjects):
            setattr(self,str("texture"+str(i)+"_col"), self.texture.data[i:i+1, 0:3, ...] )


        self.features = self.sampled_texture
        self.fake = self.netG(self.features)

        if not self.opt.target_downsample_factor == 1: # render internally at higher res
            self.fake = F.interpolate(self.fake, scale_factor=1/self.opt.target_downsample_factor, mode="bilinear", align_corners=True)     


    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        #fake_AB = self.fake_AB_pool.query(torch.cat((self.input_d, self.fake), 1))
        #fake_AB = self.fake_AB_pool.query(self.fake)
        #fake_AB = self.fake
        fake_AB = torch.cat((self.input_d, self.fake), 1)
        #print(fake_AB.shape)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.input_d, self.target), 1)
        #real_AB = self.target
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        if self.loss_D > self.opt.tld: 
            self.loss_D.backward()

    def criterionVGG(self, fake, target):
        vgg_fake = self.vgg(fake)
        vgg_target = self.vgg(target)

        content_weight = 1.0
        style_weight = 1.0

        content_loss = self.criterionL2(vgg_target.relu2_2, vgg_fake.relu2_2)

        # gram_matrix
        gram_style = [gram_matrix(y) for y in vgg_target]
        style_loss = 0.0
        for ft_y, gm_s in zip(vgg_fake, gram_style):
            gm_y = gram_matrix(ft_y)
            style_loss += self.criterionL2(gm_y, gm_s)

        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss


    def backward_G(self, epoch, apply_grad=True):
        fake_weight = 1
        self.loss_G_GAN = torch.zeros([1]).float().to(self.device)
        if self.use_gan and epoch > self.opt.suspend_gan_epochs:
            #First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.input_d, self.fake), 1)
            #fake_AB = self.fake
            pred_fake = self.netD(fake_AB)
            #self.loss_G_GAN = fake_weight * self.criterionGAN(pred_fake, True) * 0.0#0.1 ##<<<<<
            self.loss_G_GAN = fake_weight * self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
       
        # Second, G(A) = B
        if self.opt.lossType == 'L1':
            self.loss_G_total = self.loss_G_L1 = fake_weight * self.criterionL1(self.fake, self.target) * self.opt.lambda_L1
        elif self.opt.lossType == 'VGG':
            self.loss_G_total = self.loss_G_VGG = fake_weight * self.criterionVGG(self.fake, self.target) * self.opt.lambda_VGG# vgg loss is quite high
        elif self.opt.lossType == 'all':
            self.loss_G_L1 = fake_weight * self.criterionL1(self.fake, self.target) * self.opt.lambda_L1
            self.loss_G_VGG= fake_weight * self.criterionVGG(self.fake, self.target) * self.opt.lambda_VGG
            self.loss_G_total = self.loss_G_L1 + self.loss_G_VGG
        elif self.opt.lossType == 'GAN': 
            self.loss_G_L1 = fake_weight * self.criterionL1(self.fake, self.target) * self.opt.lambda_L1
            self.loss_G_total = self.loss_G_L1 + self.loss_G_GAN
        else:
            self.loss_G_total = self.loss_G_L1 = fake_weight * self.criterionL2(self.fake, self.target) * self.opt.lambda_L1

        # col tex loss
        #self.loss_G_L1 += texture_weight * self.criterionL1(self.sampled_texture_col, self.target) * self.opt.lambda_L1


        # regularizer of texture
        self.regularizerTex = 0.0
        regularizerWeight = 1.0#texture_weight
        if self.opt.hierarchicalTex:   
            ## backup
            #high_weight=80.0
            #medium_weight=20.0
            #low_weight=10.0
            #lowest_weight=0.0
            ###
            high_weight=8.0#4
            medium_weight=2.0
            low_weight=1.0
            lowest_weight=0.0
            ####
            self.regularizerTex += regularizerWeight * torch.mean(torch.pow( self.texture.high_level_tex, 2.0 )  ) * high_weight
            self.regularizerTex += regularizerWeight * torch.mean(torch.pow( self.texture.medium_level_tex, 2.0 )) * medium_weight
            self.regularizerTex += regularizerWeight * torch.mean(torch.pow( self.texture.low_level_tex, 2.0 )   ) * low_weight
            self.regularizerTex += regularizerWeight * torch.mean(torch.pow( self.texture.lowest_level_tex, 2.0 )   ) * lowest_weight

        #self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.regularizerTex
        self.loss_G = self.loss_G_total + self.regularizerTex

        if(apply_grad):
            self.loss_G.backward()

    def optimize_parameters(self, epoch_iter):
        self.forward()

        # self.optimizer_T.zero_grad()

        # ## loss = L1(texture - target) 
        # self.loss_L1 = self.criterionL1(self.sampled_texture_col, self.target)
        # self.loss_L1.backward()
        # self.optimizer_T.step()
        if self.trainRenderer:
            # update Discriminator
            if self.use_gan:
        
                self.set_requires_grad(self.netD, True)
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()

                # # update Generator
                self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.optimizer_T.zero_grad()

            self.backward_G(epoch_iter)

            self.optimizer_G.step()
            self.optimizer_T.step()

        else:
            # update texture
            self.optimizer_T.zero_grad()
            if self.use_gan:
                self.backward_D()
            self.backward_G(epoch_iter)
            self.optimizer_T.step()

    def compute_losses(self): 
        self.backward_G(0, apply_grad=False)