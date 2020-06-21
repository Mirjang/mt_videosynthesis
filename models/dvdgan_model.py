import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import functools
import random
from .networks import VGG16, UnetSkipConnectionBlock, ConvLSTMCell, ConvGRUCell
from .networks import NLayerDiscriminator

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


class VideoDiscriminatorNet(nn.Module): 
    def __init__(self, nframes, image_nc=3):
        self.nframes = nframes
        self.image_nc = image_nc
        self.hidden_dims = hidden_dims  

        def conv_relu(in_dims, out_dims, stride = 1): 
            layer = [nn.Conv2d(in_dims, out_dims, kernel_size=3, bias=True, padding=1, stride=stride, padding_mode="reflect")]
            layer += [nn.LeakyReLU(0.2)]
            return layer  


        def conv3d_relu_x2(in_dims, out_dims, stride = 1): 
            layer=[]
            layer += [nn.Conv3d(in_dims, in_dims//2, kernel_size=(3,3,3), stride=1, padding=0)]
            layer += [nn.LeakyReLU(0.2)]
            layer += [nn.Conv3d(in_dims//2, out_dims, kernel_size=(3,3,3), stride=1, padding=0)]
            layer += [nn.LeakyReLU(0.2)]
            layer += [nn.AvgPool3d(kernel_size=2, stride=2)]
            return layer


        encoder = []
        encoder += [conv3d_relu_x2(nframes, nframes//4)]
        encoder += [conv3d_relu_x2(nframes//4, nframes//16)]
        encoder += [nn.Sigmoid()]

        self.encoder = nn.Sequential(*encoder)


    def forward(self, x): 
        return self.encoder(x)


class GRUEncoderDecoderNet(nn.Module): 
    def __init__(self, nframes, image_nc=3, hidden_dims = 16):
        super(GRUEncoderDecoderNet, self).__init__()
        self.nframes = nframes
        self.image_nc = image_nc
        self.hidden_dims = hidden_dims  

        encoder = []

        def conv_relu(in_dims, out_dims, stride = 1): 
            layer = [nn.Conv2d(in_dims, out_dims, kernel_size=3, bias=True, padding=1, stride=stride, padding_mode="reflect")]
            layer += [nn.LeakyReLU(0.2)]
            return layer

        encoder += conv_relu(image_nc,16, stride = 2)
        encoder += conv_relu(16,32, stride = 2)
        encoder += conv_relu(32,64, stride = 1)
        encoder += conv_relu(64,hidden_dims)


        self.encoder = nn.Sequential(*encoder)

        enc2hidden = conv_relu(hidden_dims,hidden_dims)
        enc2cell = conv_relu(hidden_dims,hidden_dims)
        
        self.enc2hidden = nn.Sequential(*enc2hidden)

        self.gru = ConvGRUCell(hidden_dims, hidden_dims, (3,3), True)
        decoder = []
        decoder += [nn.Upsample(scale_factor=2)]
        decoder += conv_relu(hidden_dims,32)
        decoder += conv_relu(32,16)

        decoder += [nn.Upsample(scale_factor=2)]
        decoder += [nn.Conv2d(16, image_nc, kernel_size=3, stride=1, bias=True, padding=1, padding_mode="reflect")]
        decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)



    def forward(self, x): 
        x = x * 2 - 1 #[0,1] -> [-1,1]
        if len(x.shape) == 4: 
            x = x.unsqueeze(1)

        N,T,C,H,W = x.shape
        out = torch.zeros((N,self.nframes,C,H,W), device = x.device)
        out[:,0] = x[:,0,...]
        h = None

        for i in range(1,self.nframes):
            
            if i<=T: # frame was provided as input
                x_i = x[:,i-1,...]

            x_i = self.encoder(x_i)

            if h is None: 
                h = self.enc2hidden(x_i)

            h = self.gru(x_i, h)
            x_i = self.decoder(h)
            out[:,i] = x_i
        #x = torch.reshape(x, (N,self.nframes,C,H,W))
        out = (out+1) / 2.0 #[-1,1] -> [0,1] for vis 
        return out




class DvdGanModel(BaseModel):
    def name(self):
        return 'DvdGanModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        #parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(norm='instance', netG='unet_256')
        #parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_S', type=float, default=1.0, help='weight for spatial loss')
            parser.add_argument('--lambda_T', type=float, default=1.0, help='weight for temporal loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.num_display_frames = min(opt.num_display_frames, int(opt.max_clip_length*opt.fps)-1)
        self.nframes = int(opt.max_clip_length * opt.fps / opt.skip_frames)
        self.opt = opt
        self.ndsframes = opt.dvd_spatial_frames
        assert self.nframes > self.ndsframes+1, "number of frames sampled for disc should be leq to number of total frames generated (length-1)"
        print(self.device)  

        self.train_range = (1,self.nframes)
        if self.opt.train_mode is "frame":
            self.train_range = (1,1)
        elif self.opt.train_mode is "video":
            self.train_range = (self.nframes,self.nframes)
        elif self.opt.train_mode is "mixed":
            self.train_range = (1,self.nframes)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ["prediction_target_video", "lossperframe_plt"]

        self.lossperframe_plt = {"opts": {
                    'title': "Loss per frame",
                    #'legend': ["L1 loss per frame"],
                    'xlabel': 'frame',
                    'ylabel': 'loss',
                    } ,
          #          "Y":np.array((1,1)), "X":np.array((1,1))
            }

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

        netG = GRUEncoderDecoderNet(self.nframes,opt.input_nc, hidden_dims=128)
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:

            self.netDs = NLayerDiscriminator(3, ndf = 32, n_layers = 3, use_sigmoid=True)
            self.netDt = VideoDiscriminatorNet(self.nframes, 3)
            # define loss functions
            self.criterionGAN = networks.GANLoss(False).to(self.device)

            # initialize optimizers
            self.optimizers = []
            #self.optimizer = torch.optim.SGD(self.netG.parameters(), opt.lr)
            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizerG)

            self.optimizerDs = torch.optim.Adam(self.netDs.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizerDs)            
            
            self.optimizerDt = torch.optim.Adam(self.netDt.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizerDt)

    def set_input(self, input):
        self.target_video = input['VIDEO'].to(self.device).permute(0,1,4,2,3).float() / 255.0 #normalize to [0,1] for vis and stuff 
        self.input = self.target_video[:,0,...]#first frame
        _, T, *_ = self.target_video.shape
        self.target_video = self.target_video[:, :min(T,self.nframes),...]

    def forward(self, frame_length = -1, train = True):
        if hasattr(self.netG, "nFrames"):
            self.netG.nFrames = frame_length if frame_length>0 else self.nframes
        if train:
            video = self.netG(self.target_video[:,:random.randrange(*self.train_range),...])
        else:
            video = self.netG(self.input)
        self.predicted_video = video
        self.prediction_target_video = torch.cat([self.predicted_video, self.target_video], dim = 4)

        for i in range(self.num_display_frames//2):
            setattr(self,f"frame_{i}", self.predicted_video[:,i,...] )
        for i in range(self.num_display_frames//2):
            ith_last = self.num_display_frames//2 -i +1
            setattr(self,f"frame_{i + self.num_display_frames//2}", self.predicted_video[:,-ith_last,...] )
        video = video * 256
        return video.permute(0,1,3,4,2)


    def epoch_frame_length(self, epoch): 
        increase_intervals = 10
        iter_per_interval = 20
        return max(8,min(self.nframes // increase_intervals * (epoch // iter_per_interval +1), self.nframes))


    def backward_Ds(self):
        self.loss_Ds_fake = 0
        self.loss_Ds_real = 0
        for i in random.sample(range(1, self.nframes), self.ndsframes):

            fake,real = self.predicted_video[:,i,...].detach() , self.target_video[:,i,...]

            # Fake
            # stop backprop to the generator by detaching fake_B
            pred_fake = self.netDs(fake)
            self.loss_Ds_fake += self.criterionGAN(pred_fake, False)

            # Real
            pred_real = self.netDt(real)
            self.loss_Ds_real += self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_Ds = (self.loss_Dt_fake + self.loss_Dt_real) * 0.5 /self.ndsframes

        if self.loss_Ds > self.opt.tld: 
            self.loss_Ds.backward()

    def backward_Dt(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = self.netDs(self.predicted_video.detach())
        self.loss_Dt_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netDt(self.target_video)
        self.loss_Dt_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_Dt = (self.loss_Dt_fake + self.loss_Dt_real) * 0.5
        if self.loss_Dt > self.opt.tld: 
            self.loss_Dt.backward()


    def backward_G(self, epoch):
        # First, G(A) should fake the discriminator

        self.loss_Gs = 0
        for i in random.sample(range(1, self.nframes), self.ndsframes):
            pred_fake_ds = self.netDs(self.predicted_video[:,i,...])

            self.loss_Gs += self.criterionGAN(pred_fake_ds, True)

        self.loss_Gs = self.loss_Gs / self.ndsframes

        pred_fake_dt = self.netDt(self.predicted_video)
        self.loss_Gt = self.criterionGAN(pred_fake_dt, True)
        self.loss_G = (self.loss_Gs + self.loss_Gt) * 0.5

        self.loss_G.backward()

    def optimize_parameters(self, epoch_iter):
        Te = self.epoch_frame_length(epoch_iter)
        self.forward(frame_length=Te)

        self.optimizer.zero_grad()
        _, T,*_ = self.predicted_video.shape
        _, TT,*_ = self.target_video.shape
   
        #T = min(T,TT,Te) # just making sure to cut target if we didnt predict all the frames and to cut prediction, if we predicted more than target (i.e. we already messed up somewhere)
        
        # update Discriminator(s)
        self.set_requires_grad(self.netDs, True)
        self.optimizer_Ds.zero_grad()
        self.backward_Ds()
        self.optimizer_Ds.step()

        self.set_requires_grad(self.netDt, True)
        self.optimizer_Dt.zero_grad()
        self.backward_Dt()
        self.optimizer_Dt.step()


        # update Generator
        self.set_requires_grad(self.netDs, False)
        self.optimizer_G.zero_grad()

        self.backward_G(epoch_iter)

        self.optimizer_G.step()


    # def compute_losses(self, secs = 2, fps = 30): 
    #     T = secs * fps *1.0
    #     with torch.no_grad():
    #         self.netG.eval()
    #         self.forward(frame_length=T, train=False)
    #         self.netG.train()

    #         _, T,*_ = self.predicted_video.shape
    #         _, TT,*_ = self.target_video.shape

    #         # print(torch.min(self.predicted_video), torch.max(self.predicted_video))
    #         # print(torch.min(self.target_video), torch.max(self.target_video))

    #         T = min(T,TT) # just making sure to cut target if we didnt predict all the frames and to cut prediction, if we predicted more than target (i.e. we already messed up somewhere)
    #         ## loss = L1(prediction - target) 
    #         #print(torch.min(self.target_video), torch.max(self.target_video))
    #         loss_L1 = torch.zeros([1], device = self.device)
    #         lpf=[]
    #         for i in range(1,T):
    #             l_i = self.criterionL1(self.predicted_video[:,i,...], self.target_video[:,i,...])
    #             loss_L1 += l_i
    #             lpf.append(l_i.item())
    #         self.loss_L1 = sum(lpf)
    #     self.lossperframe_plt["Y"] = lpf
    #     self.lossperframe_plt["X"] =list(range(1, len(lpf)+1))

    #     return loss_L1, lpf