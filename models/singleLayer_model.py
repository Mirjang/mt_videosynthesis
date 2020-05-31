import torch
import torch.nn as nn
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
        use_norm = True

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

    def forward(self, feature_map, expressions):
        # TODO incorporat expressions
        return self.model(feature_map)

def define_Renderer(renderer, n_feature,ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm)
    N_OUT = 3
    #net = UnetRenderer(N_FEATURE, N_OUT, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    net = UnetRenderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return networks.init_net(net, init_type, init_gain, gpu_ids)


class Texture(nn.Module):
    def __init__(self, n_textures, n_features, dimensions, device):
        super(Texture, self).__init__()
        #self.register_parameter('data', torch.nn.Parameter(torch.randn(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True)))
        #self.register_parameter('data', torch.nn.Parameter(2.0 * torch.ones(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True) -1.0))
        self.register_parameter('data', torch.nn.Parameter(torch.zeros(n_textures, n_features, dimensions, dimensions, device=device, requires_grad=True)))

    def forward(self, uv_inputs, texture_id):
        uvs = torch.stack([uv_inputs[:,0,:,:], uv_inputs[:,1,:,:]], 3)
        return torch.nn.functional.grid_sample(self.data[texture_id:texture_id+1, :, :, :], uvs, mode='bilinear', padding_mode='border')

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



class SingleLayerModel(BaseModel):
    def name(self):
        return 'SingleLayerModel'

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


        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['L1']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['texture_col', 'sampled_texture_col','target']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['texture']
        else:  # during test time, only load Gs
            self.model_names = ['texture']


        # load/define networks
        #self.netG = define_Renderer(opt.rendererType, opt.tex_features + 3, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #self.netG = define_Renderer(opt.rendererType, opt.tex_features+2, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)#<<<<<<<<<<<<<<<<

        # texture
        
        self.texture = define_Texture(opt.nObjects, opt.tex_features, opt.tex_dim, device=self.device, gpu_ids=self.gpu_ids)
       
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_T = torch.optim.SGD(self.texture.parameters(), opt.lr)
            #self.optimizer_T = torch.optim.Adam(self.texture.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_T)



    def set_input(self, input):
        self.target = input['TARGET'].to(self.device)
        self.input_uv = input['UV'][:,:2,:,:].to(self.device)
        self.input_mask = input['MASK'][:,:1,:,:].to(self.device)
        self.image_paths = input['paths']
        self.object_id = 0


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


    def maskErosion(self, mask):
        offsetY = int(self.opt.erosionFactor * 40)
        # throat
        mask2 = mask[:,:,0:-offsetY,:]
        mask2 = torch.cat([torch.ones_like(mask[:,:,0:offsetY,:]), mask2], 2)
        #return mask * mask2
        # forehead
        mask3 = mask[:,:,offsetY:,:]
        mask3 = torch.cat([mask3, torch.ones_like(mask[:,:,0:offsetY,:])], 2)
        mask = mask * mask2 * mask3 

        offsetX = int(self.opt.erosionFactor * 15)
        # left
        mask4 = mask[:,:,:,0:-offsetX]
        mask4 = torch.cat([torch.ones_like(mask[:,:,:,0:offsetX]), mask4], 3)
        # right
        mask5 = mask[:,:,:,offsetX:]
        mask5 = torch.cat([mask5,torch.ones_like(mask[:,:,:,0:offsetX])], 3)
        return mask * mask4 * mask5

    def forward(self):
        self.sampled_texture = self.texture(self.input_uv, self.object_id)
        self.sampled_texture_col = self.sampled_texture[:,0:3,:,:]
        self.texture_col = self.texture.data[self.object_id:self.object_id+1,0:3,:,:]
        #self.features = self.sh_Layer(self.sampled_texture, self.extrinsics)
        #features = torch.cat([self.input_uv[:,0:2,:,:], features], 1) #<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # add background from the target as input
        #mask = (self.input_uv[:,0:1,:,:] == INVALID_UV) & (self.input_uv[:,1:2,:,:] == INVALID_UV)
        mask = self.input_mask == 0
        #mask = self.maskErosion(mask)
        self.mask = torch.cat([mask,mask,mask], 1)
        #self.background = torch.where(mask, self.target, torch.zeros_like(self.target))
        #self.features = torch.cat([self.features, self.background], 1)

        #self.background = mask

        #self.fake = self.netG(self.features, self.expressions)







    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.input_uv, self.fake), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.input_uv, self.target), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def computeEpochWeight(self, epoch):
        # adaptive weighting scheme
        fw = 1.0
        tw = 10.0 # -> scales lr

        minIter = 15
        maxIter = 30
        
        alpha = (epoch - minIter) / (maxIter - minIter)
        if epoch < minIter:
            fw *= 0.0
            tw *= 1.0
        elif epoch < maxIter:
            fw *= alpha
            tw *= (1.0 - alpha)
        else:
            fw *= 1.0
            tw *= 0.0

        fw += 0.1
        tw += 0.25 # <<<

        return (fw, tw)

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


    def backward_G(self, epoch):
        # compute epoch weight
        (fake_weight, texture_weight) = self.computeEpochWeight(epoch)

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.input_uv, self.fake), 1)
        pred_fake = self.netD(fake_AB)
        #self.loss_G_GAN = fake_weight * self.criterionGAN(pred_fake, True) * 0.0#0.1 ##<<<<<
        self.loss_G_GAN = fake_weight * self.criterionGAN(pred_fake, True) * 0.01
       

        # Second, G(A) = B
        if self.opt.lossType == 'L1':
            self.loss_G_L1 = fake_weight * self.criterionL1(self.fake, self.target) * self.opt.lambda_L1
        elif self.opt.lossType == 'VGG':
            self.loss_G_L1 = fake_weight * self.criterionVGG(self.fake, self.target) * self.opt.lambda_L1 * 0.001 # vgg loss is quite high
        else:
            self.loss_G_L1 = fake_weight * self.criterionL2(self.fake, self.target) * self.opt.lambda_L1

        # col tex loss
        self.loss_G_L1 += texture_weight * self.criterionL1(self.sampled_texture_col, self.target) * self.opt.lambda_L1


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

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.regularizerTex

        self.loss_G.backward()

    def optimize_parameters(self, epoch_iter):
        self.forward()

        self.optimizer_T.zero_grad()

        ## loss = L1(texture - target) 
        self.loss_L1 = self.criterionL1(self.texture_col, self.target)
        self.loss_L1.backward()
        self.optimizer_T.step()
        # if self.trainRenderer:
        #     # update Discriminator
        #     self.set_requires_grad(self.netD, True)
        #     self.optimizer_D.zero_grad()
        #     self.backward_D()
        #     self.optimizer_D.step()

        #     # update Generator
        #     self.set_requires_grad(self.netD, False)
        #     self.optimizer_G.zero_grad()
        #     self.optimizer_T.zero_grad()

        #     self.backward_G(epoch_iter)

        #     self.optimizer_G.step()
        #     self.optimizer_T.step()

        # else:
        #     # update texture
        #     self.optimizer_T.zero_grad()
        #     self.backward_D()
        #     self.backward_G(epoch_iter)
        #     self.optimizer_T.step()