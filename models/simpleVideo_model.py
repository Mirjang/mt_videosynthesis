import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import functools

from .networks import VGG16, UnetSkipConnectionBlock

################
###  HELPER  ###
################

INVALID_UV = -1.0


from torchvision import models
from collections import namedtuple


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class ConvLiftNet(nn.Module): 
    def __init__(self, nframes, image_nc=3):
        super(ConvLiftNet, self).__init__()
        self.nframes = nframes
        self.image_nc = image_nc
        model = [nn.Conv2d(image_nc, image_nc*nframes, kernel_size=3, stride=1, bias=True, padding=1, padding_mode="mirror")]
        model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x): 
        N,C,H,W = x.shape
        y = self.model(x)
        y = torch.reshape(y, (N,self.nframes,C,H,W))
        return y



class SimpleVideoModel(BaseModel):
    def name(self):
        return 'SimpleVideoModel'

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
        self.visual_names = ["predicted_video", "target_video"]

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG']
        else:  # during test time, only load Gs
            self.model_names = ['netG']

        # load/define networks
        #self.netG = define_Renderer(opt.rendererType, opt.tex_features + 3, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #self.netG = define_Renderer(opt.rendererType, opt.tex_features+2, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)#<<<<<<<<<<<<<<<<

        netG = ConvLiftNet(30,3)
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            #self.optimizer = torch.optim.SGD(self.texture.parameters(), opt.lr)
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)



    def set_input(self, input):
        self.target = input['VIDEO'].to(self.device).permute(0,1,4,2,3) / 256.0 #normalize to [0,1]
        self.input = self.target[:,0,...] #first frame
        self.target_video = self.target[0, :150,...]


    def forward(self):
        video = self.netG(self.input)
        self.predicted_video = video
        video = video * 256
        return video.permute(0,1,3,4,2)


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

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.input_uv, self.fake), 1)
        pred_fake = self.netD(fake_AB)
        #self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.0#0.1 ##<<<<<
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.01
       

        # Second, G(A) = B
        if self.opt.lossType == 'L1':
            self.loss_G_L1 =  self.criterionL1(self.fake, self.target) * self.opt.lambda_L1
        elif self.opt.lossType == 'VGG':
            self.loss_G_VGG = self.criterionVGG(self.fake, self.target) * self.opt.lambda_L1 * 0.001 # vgg loss is quite high
        else:
            self.loss_G_L2 = self.criterionL2(self.fake, self.target) * self.opt.lambda_L1

        # col tex loss
        self.loss_G_L1 += self.criterionL1(self.sampled_texture_col, self.target) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.regularizerTex

        self.loss_G.backward()

    def optimize_parameters(self, epoch_iter):
        self.forward()

        self.optimizer.zero_grad()
        _, T,_,_,_ = self.predicted_video.shape
        ## loss = L1(texture - target) 
        target = self.target[:,:T,...]
    
        self.loss_L1 = self.criterionL1(self.predicted_video, target)
        self.loss_L1.backward()
        self.optimizer.step()
        # if self.trainRenderer:
        #     # update Discriminator
        #     self.set_requires_grad(self.netD, True)
        #     self.optimizer_D.zero_grad()
        #     self.backward_D()
        #     self.optimizer_D.step()

        #     # update Generator
        #     self.set_requires_grad(self.netD, False)
        #     self.optimizer_G.zero_grad()


        #     self.backward_G(epoch_iter)

        #     self.optimizer_G.step()