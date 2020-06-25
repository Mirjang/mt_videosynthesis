import torch
import torch.nn as nn
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel
from . import networks
import numpy as np
import functools
import random
from .networks import VGG16, UnetSkipConnectionBlock, ConvLSTMCell, ConvGRUCell
from .networks import NLayerDiscriminator
from collections import OrderedDict
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
    def __init__(self, image_nc=3, ndf = 32):
        super(VideoDiscriminatorNet, self).__init__()
        self.image_nc = image_nc

        def conv3d_relu_x2(in_dims, out_dims, stride = 1): 
            layer=[]
            layer += [nn.Conv3d(in_dims, in_dims*2, kernel_size=(3,3,3), stride=stride, padding=0, padding_mode="replicate")]
            layer += [nn.LeakyReLU(0.2)]
            layer += [nn.Conv3d(in_dims*2, out_dims, kernel_size=(3,3,3), stride=stride, padding=0, padding_mode="replicate")]
            layer += [nn.LeakyReLU(0.2)]
            return layer

        encoder = []
        encoder += [nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))]
        encoder += [nn.Conv3d(image_nc, ndf, kernel_size=1, stride=1, padding=0, padding_mode="replicate")]
        encoder += [nn.LeakyReLU(0.2)]
        encoder += conv3d_relu_x2(ndf, ndf*4, stride = (1,2,2))
        encoder += [nn.Conv3d(ndf*4, 1, kernel_size=1, stride=1, padding=0, padding_mode="replicate")]
        encoder += [nn.Sigmoid()]

        self.encoder = nn.Sequential(*encoder)


    def forward(self, x): 
        x = x * 2 - 1 #[0,1] -> [-1,1]

        x = x.permute(0,2,1,3,4) # flip time(1) and color channels (2)
        x = self.encoder(x).squeeze()
        # print(f"df: {x.shape}")
        return x

class SpatialDiscriminatorNet(nn.Module): 
    def __init__(self, image_nc, ndf = 16, mlp_in_dims = 8):
        super(SpatialDiscriminatorNet, self).__init__()

        self.image_nc = image_nc
        def conv_relu_x2(in_dims, out_dims, stride = 1): 
            layer = [nn.Conv2d(in_dims, in_dims*2, kernel_size=3, bias=True, padding=0, stride=stride, padding_mode="reflect")]
            layer += [nn.LeakyReLU(0.2)]
            layer += [nn.Conv2d(in_dims*2, out_dims, kernel_size=3, bias=True, padding=0, stride=stride, padding_mode="reflect")]
            layer += [nn.LeakyReLU(0.2)]
            return layer  

        encoder = []
        encoder += [nn.Conv2d(image_nc, ndf, kernel_size=3, bias=True, stride = 1)]
        encoder += [nn.LeakyReLU(0.2)]
        encoder += conv_relu_x2(ndf, ndf*4)

        encoder += conv_relu_x2(ndf*4, ndf*8, stride = 2)
        encoder += [nn.Conv2d(ndf*8, 1, kernel_size=1, stride = 2, bias=True)]
        encoder += [nn.Sigmoid()]
      #  encoder += [nn.Upsample(size =(mlp_in_dims,mlp_in_dims))]

        self.encoder = nn.Sequential(*encoder)
        # self.mlp_shape = mlp_in_dims**2 * 1

        # mlp = []
        # mlp += [nn.Linear(self.mlp_shape, 1)]
        # # mlp += [nn.ReLU()]
        # # mlp += [nn.Linear(64,1)]
        # mlp += [nn.Sigmoid()]
        # self.mlp = nn.Sequential(*mlp)

    def forward(self, x): 
        x = x * 2 - 1 #[0,1] -> [-1,1]

        x = self.encoder(x)
        # x = x.view(-1, self.mlp_shape)
        # x = self.mlp(x)
        #print(f"ds: {x.shape}")
        return x


class GRUEncoderDecoderNet(nn.Module): 
    def __init__(self, nframes, image_nc=3, ngf = 32, hidden_dims = 16, enc2hidden = False):
        super(GRUEncoderDecoderNet, self).__init__()
        self.nframes = nframes
        self.image_nc = image_nc
        self.hidden_dims = hidden_dims  

        encoder = []

        def conv_relu(in_dims, out_dims, stride = 1): 
            layer = [nn.Conv2d(in_dims, out_dims, kernel_size=3, bias=True, padding=1, stride=stride, padding_mode="reflect")]
            layer += [nn.LeakyReLU(0.2)]
            return layer

        encoder += conv_relu(image_nc,ngf, stride = 1)
        encoder += conv_relu(ngf,ngf*2, stride = 1)
      #  encoder += [nn.BatchNorm2d(32)]
      #  encoder += conv_relu(32,64, stride = 1)
      #  encoder += conv_relu(ngf*2,ngf*4)


        self.encoder = nn.Sequential(*encoder)
        
        if enc2hidden:
            enc2hidden = conv_relu(ngf*2,hidden_dims)
            self.enc2hidden = nn.Sequential(*enc2hidden)
        else: 
            self.enc2hidden = None

        self.gru = ConvGRUCell(ngf*2, hidden_dims, (3,3), True)
        decoder = []
        #decoder += [nn.Upsample(scale_factor=2)]
        decoder += conv_relu(hidden_dims,ngf)
        decoder += conv_relu(ngf,ngf//2)
        decoder += conv_relu(ngf//2,ngf//4)

        #decoder += [nn.Upsample(scale_factor=2)]
        decoder += [nn.Conv2d(ngf//4, image_nc, kernel_size=3, stride=1, bias=True, padding=1, padding_mode="reflect")]
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
                if self.enc2hidden: 
                    h = self.enc2hidden(x_i)
                else: 
                    _,_,H_i, W_i = x_i.shape
                    h = torch.zeros((N,self.hidden_dims,H_i,W_i), device = x.device)



            h = self.gru(x_i, h)
            x_i = self.decoder(h)
            out[:,i] = x_i
        #x = torch.reshape(x, (N,self.nframes,C,H,W))
        out = (out+1) / 2.0 #[-1,1] -> [0,1] for vis 
        return out



class GRUDeltaNet(nn.Module): 
    def __init__(self, nframes, image_nc=3, hidden_dims = 16):
        super(GRUDeltaNet, self).__init__()
        self.nframes = nframes
        self.image_nc = image_nc
        self.hidden_dims = hidden_dims  


        def conv_relu(in_dims, out_dims, stride = 1): 
            layer = [nn.Conv2d(in_dims, out_dims, kernel_size=3, bias=True, padding=1, stride=stride, padding_mode="reflect")]
            layer += [nn.ReLU()]
            return layer

        encoder = []
        encoder += conv_relu(image_nc,16, stride = 1)
        encoder += conv_relu(16,32, stride = 1)
        encoder += [nn.AvgPool2d(2,2)]
        encoder += conv_relu(32,64, stride = 1)
        encoder += conv_relu(64,hidden_dims)


        self.encoder = nn.Sequential(*encoder)

        enc2hidden = conv_relu(hidden_dims,hidden_dims)        
        self.enc2hidden = nn.Sequential(*enc2hidden)

        self.gru = ConvGRUCell(hidden_dims, hidden_dims, (3,3), True)
        decoder = []
        decoder += conv_relu(hidden_dims,32)
        decoder += [nn.Upsample(scale_factor=2)]        
        decoder += [nn.Conv2d(32, image_nc, kernel_size=3, stride=1, bias=True, padding=1, padding_mode="reflect")]
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

            e_i = self.encoder(x_i)

            if h is None: 
                h = torch.zeros_like(e_i, device = x.device)

            #h = self.gru(e_i, h)
            x_i = self.decoder(e_i)
            out[:,i] = x_i
        #x = torch.reshape(x, (N,self.nframes,C,H,W))
        out = (out+1) / 2.0 #[-1,1] -> [0,1] for vis 
        return out


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
            self.train_range = (1,2)
        elif self.opt.train_mode is "video":
            self.train_range = (self.nframes-1,self.nframes)
        elif self.opt.train_mode is "mixed":
            self.train_range = (1,self.nframes)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ["prediction_target_video", "activity_diag_plt"]#, "lossperframe_plt"]
        
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

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['Gs', 'Gt', 'accDs_real','accDs_fake','accDt_real', 'accDt_fake']
        if opt.pretrain_epochs > 0:
            self.loss_names.append('G_L1')
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG', 'netDs', 'netDt']
        else:  # during test time, only load Gs
            self.model_names = ['netG']

        # load/define networks
        netG = GRUEncoderDecoderNet(self.nframes,opt.input_nc,ngf = 32, hidden_dims=64, enc2hidden = True)
        #netG = GRUDeltaNet(self.nframes,opt.input_nc,hidden_dims=64)

        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            netDs = SpatialDiscriminatorNet(6, ndf = 16)
            self.netDs = networks.init_net(netDs, opt.init_type, opt.init_gain, self.gpu_ids)

            netDt = VideoDiscriminatorNet(3, ndf = 32)
            self.netDt = networks.init_net(netDt, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(False).to(self.device)
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
        increase_intervals = 5
        iter_per_interval = 5
        return max(8,min(self.nframes // increase_intervals * (epoch // iter_per_interval +1), self.nframes))


    def backward_Ds(self, train_threshold = 0, conditional = True):
        self.loss_Ds_fake = 0
        self.loss_Ds_real = 0
        self.loss_accDs_fake = 0
        self.loss_accDs_real = 0
        _, T, *_ = self.predicted_video.shape
        for i in random.sample(range(1, T), min(T-1,self.ndsframes)):

            fake,real = self.predicted_video[:,i,...].detach().to(self.device) , self.target_video[:,i,...]

            if conditional: 
                fake = torch.cat([self.input, fake], dim = 1)
                real = torch.cat([self.input, real], dim = 1)
            # Fake
            # stop backprop to the generator by detaching fake_B
            pred_fake = self.netDs(fake)
            self.loss_accDs_fake = 1 - torch.mean(pred_fake).item()
            self.loss_Ds_fake += self.criterionGAN(pred_fake, False)
            
            # Real
            pred_real = self.netDs(real)
            self.loss_accDs_real += torch.mean(pred_real).item()
            self.loss_Ds_real += self.criterionGAN(pred_real, True)

        self.loss_accDs_real = self.loss_accDs_real / self.ndsframes
        self.loss_accDs_fake = self.loss_accDs_fake / self.ndsframes
        self.loss_acc_Ds = (self.loss_accDs_fake + self.loss_accDs_real)*.5
        self.loss_Ds_real = self.loss_Ds_real / self.ndsframes
        self.loss_Ds_fake = self.loss_Ds_fake / self.ndsframes
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

    def backward_G(self, epoch, train_threshold = 0, conditional = True):
        # First, G(A) should fake the discriminator
        self.loss_G = 0
        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0

        for i in random.sample(range(1, self.nframes), self.ndsframes):
            fake = torch.cat([self.input, self.predicted_video[:,i,...]], dim = 1)

            pred_fake_ds = self.netDs(fake)
            self.loss_Gs += self.criterionGAN(pred_fake_ds, True)
        self.loss_Gs = self.loss_Gs / self.ndsframes

        pred_fake_dt = self.netDt(self.predicted_video)
        self.loss_Gt = self.criterionGAN(pred_fake_dt, True)

        trust_Ds = 1 if self.loss_acc_Ds > train_threshold else 0
        trust_Dt = 1 if self.loss_acc_Dt > train_threshold else 0
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
        verbose = verbose or self.opt.verbose
        #Te = self.epoch_frame_length(epoch_iter)
        Te = self.nframes
        self.forward(frame_length=Te)
        _, T,*_ = self.predicted_video.shape
        _, TT,*_ = self.target_video.shape
   
        T = min(T,TT,Te) # just making sure to cut target if we didnt predict all the frames and to cut prediction, if we predicted more than target (i.e. we already messed up somewhere)
       # tld = self.opt.tld if epoch_iter > self.opt.pretrain_epochs else 0
        tld = self.opt.tld
        #update Discriminator(s)
        self.set_requires_grad(self.netDs, True)
        self.optimizer_Ds.zero_grad()
        if self.backward_Ds(train_threshold=tld):
            if verbose: 
                diagnose_network(self.netDs,"Ds")
            nn.utils.clip_grad_norm_(self.netDs.parameters(), self.opt.clip_grads)
            self.optimizer_Ds.step()

        self.set_requires_grad(self.netDt, True)
        self.optimizer_Dt.zero_grad()
        if self.backward_Dt(train_threshold=tld):
            if verbose: 
                diagnose_network(self.netDt,"Dt")
            nn.utils.clip_grad_norm_(self.netDt.parameters(), self.opt.clip_grads)
            self.optimizer_Dt.step()

        # update Generator
        self.loss_Gs = 0
        self.loss_Gt = 0
        self.loss_G_L1 = 0


        self.set_requires_grad(self.netDs, False)
        self.set_requires_grad(self.netDt, False)

        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        if self.backward_G(epoch_iter, train_threshold = self.opt.tlg):
            if verbose: 
                diagnose_network(self.netG, "netG")
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.clip_grads)
            self.optimizer_G.step()

        
        self.activity_diag = torch.cat([self.activity_diag, torch.zeros(1,len(self.activity_diag_indices))], dim = 0)
        self.activity_diag_plt["X"] = list(range(self.activity_diag.shape[0]))
        self.activity_diag_plt["Y"] = self.activity_diag


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

                fake = torch.cat([self.input, fake], dim = 1)

                dspf.append(torch.mean(self.netDs(fake)).item())
    
            ds_perframe["Y"] = dspf
            ds_perframe["X"] =list(range(len(dspf)))

            dt_fake = torch.mean(self.netDt(self.predicted_video)).item()

            return OrderedDict([("val_l1", loss_L1.item()), ("val_Ds", sum(dspf)/len(dspf)), ("val_Dt", dt_fake)]), \
                    OrderedDict([("val_pred_tar_video", self.prediction_target_video),("val_ds_per_frame_plt", ds_perframe), ])