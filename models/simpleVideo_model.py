import torch
import torch.nn as nn
from .networks import ConvLSTMCell, ConvGRUCell

class LSTMEncoderDecoderNet(nn.Module): 
    def __init__(self, nframes, image_nc=3, hidden_dims = 16):
        super(LSTMEncoderDecoderNet, self).__init__()
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
        self.enc2cell = nn.Sequential(*enc2cell)

        self.lstm = ConvLSTMCell(hidden_dims, hidden_dims, (3,3), True)
        decoder = []
        decoder += [nn.Upsample(scale_factor=2)]
        decoder += conv_relu(hidden_dims,32)
        decoder += conv_relu(32,16)

        decoder += [nn.Upsample(scale_factor=2)]
        decoder += [nn.Conv2d(16, image_nc, kernel_size=3, stride=1, bias=True, padding=1, padding_mode="reflect")]
        decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

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
        x_i = None
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
