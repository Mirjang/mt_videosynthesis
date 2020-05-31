import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import functools

from torchvision import models
from collections import namedtuple

class Texture(nn.Module):
    def __init__(self, n_textures, n_features, dimensions, device):
        super(Texture, self).__init__()
        self.device = device
        self.n_textures = n_textures
        #self.register_parameter('data', torch.nn.Parameter(torch.randn(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True)))
        #self.register_parameter('data', torch.nn.Parameter(2.0 * torch.ones(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True) -1.0))
        self.register_parameter('data', torch.nn.Parameter(torch.zeros(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True)))

    def forward(self, uv_inputs, mask_inputs):
        layers = []
        N, n_layers, H, W =mask_inputs.shape
        _, F, *_ = self.data.shape

        for layer in range(n_layers): 
            layer_idx = 2*layer
            mask_layer = mask_inputs[:,layer,:,:]
            uvs = torch.stack([uv_inputs[:,layer_idx,:,:], uv_inputs[:,layer_idx+1,:,:]], 3)

            layer_tex = torch.zeros((N,F,H,W), device=self.device)
            objects_in_mask = torch.unique(mask_layer)
            #for texture_id in range(self.n_textures): 
            for texture_id in range(self.n_textures): 
                #background is 0 in mask and has no texture atm
                mask = mask_layer == texture_id
                sample = torch.nn.functional.grid_sample(self.data[texture_id:texture_id+1, :, :, :], uvs, mode='bilinear', padding_mode='border')

                layer_tex = layer_tex + sample * mask.float()

            layers.append(layer_tex)
        return torch.cat(layers, 1)

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


def define_Texture(n_textures, n_features, dimensions, device, gpu_ids=[]):
    tex = Texture(n_textures, n_features, dimensions, device)

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

class DebugModel(BaseModel):
    def name(self):
        return 'DebugModel'

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

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.n_layers = opt.num_depth_layers
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['L1', 'dummy']
        self.loss_dummy = 0

        self.visual_names = []
        self.nObjects = opt.nObjects
        for i in range(self.nObjects):
            self.visual_names.append(str("texture"+str(i)+"_col"))

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names += ['sampled_texture_col', 'output','target']


        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['texture']
        else:  # during test time, only load Gs
            self.model_names = ['texture']

        # texture
        self.tex_features = opt.tex_features
        self.texture = define_Texture(self.nObjects, opt.tex_features, opt.tex_dim, device=self.device, gpu_ids=self.gpu_ids)
       
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []

            self.optimizer_T = torch.optim.Adam(self.texture.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_T)



    def set_input(self, input):
        self.target = input['TARGET'].to(self.device)
        self.input_uv = input['UV'].to(self.device)
        self.input_mask = input['MASK'].to(self.device)
        #self.image_paths = input['paths']


    def sh_Layer(self, tex, extrinsics):
        if self.opt.no_spherical_harmonics:
            return tex
        else:
            viewDir = [extrinsics[0][2], extrinsics[1][2], extrinsics[2][2]  ]
            basis = torch.from_numpy(spherical_harmonics_basis(viewDir)).to(self.device)
            return torch.cat([  tex[:,0:3,:,:],
                                basis[0] * tex[:,3+0:3+1,:,:], 
                                basis[1] * tex[:,3+1:3+2,:,:], basis[2] * tex[:,3+2:3+3,:,:], basis[3] * tex[:,3+3:3+4,:,:], 
                                basis[4] * tex[:,3+4:3+5,:,:], basis[5] * tex[:,3+5:3+6,:,:], basis[6] * tex[:,3+6:3+7,:,:], basis[7] * tex[:,3+7:3+8,:,:], basis[8] * tex[:,3+8:3+9,:,:],
                                tex[:,12:,:,:]
                                ], 1)

    def forward(self):
        self.sampled_texture = self.texture(self.input_uv, self.input_mask)

        #first layer first 3 channels, rgb channels for nth layers will be [:, nFeatures*n:nFeatures*n+1, ...]
        self.sampled_texture_col = self.sampled_texture[:,0:3,:,:]

        for i in range(self.nObjects):
            setattr(self,str("texture"+str(i)+"_col"), self.texture.data[i:i+1, 0:3, ...] )
        # self.texture0_col = self.texture.data[0:1,0:3,:,:]
        # self.texture1_col = self.texture.data[1:2,0:3,:,:]
        # self.texture2_col = self.texture.data[2:3,0:3,:,:]

        # simple blending, assumes tex_dims = 3
        alpha = .5
        F = self.tex_features
        output = self.sampled_texture[:, -F:,...]
        for d in reversed(range(self.n_layers-1)): 
            output = (1-alpha)* output  + alpha * self.sampled_texture[:, F*d:F*(d+1), ...] 
        self.output = output[:, 0:3, ...]

    def optimize_parameters(self, epoch_iter):
        self.forward()

        self.optimizer_T.zero_grad()
        ## loss = L1(texture - target) 
        self.loss_L1 = self.criterionL1(self.output, self.target)
        #self.loss_L1 = self.criterionL1(self.texture0_col, self.target)

        self.loss_L1.backward()
        self.optimizer_T.step()

