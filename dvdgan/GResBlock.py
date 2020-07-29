import torch
import torch.nn as nn
from torch.nn import functional as F

from .Normalization import ConditionalNorm, SpectralNorm
#import torch.nn.utils.spectral_norm as SpectralNorm
#def SpectralNorm(x): 
#    return x


class GResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=None,
                 padding=1, stride=1, n_class=96, bn=True,
                 activation=F.relu, upsample_factor=2, downsample_factor=1):
        super().__init__()

        self.upsample_factor = upsample_factor if downsample_factor is 1 else 1
        self.downsample_factor = downsample_factor
        self.activation = activation
        self.bn = bn if downsample_factor is 1 else False
        self.cbn = bn and n_class > 1

        if kernel_size is None:
            kernel_size = [3, 3]

        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))

        self.skip_proj = True
        self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))

        # if in_channel != out_channel or upsample_factor or downsample_factor:
        #     self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
        #     self.skip_proj = True

        if bn:
            if self.cbn:
                self.CBNorm1 = ConditionalNorm(in_channel, n_class) # TODO 2 x noise.size[1]
                self.CBNorm2 = ConditionalNorm(out_channel, n_class)
            else: 
                self.CBNorm1 = nn.BatchNorm2d(in_channel)
                self.CBNorm2 = nn.BatchNorm2d(out_channel)

    def forward(self, x, condition=None):

        # The time dimension is combined with the batch dimension here, so each frame proceeds
        # through the blocks independently
        BT, C, W, H = x.size()
        out = x

        if self.bn:
            out = self.CBNorm1(out, condition) if self.cbn else self.CBNorm1(out)

        out = self.activation(out)

        if self.upsample_factor != 1:
            out = F.interpolate(out, scale_factor=self.upsample_factor)

        out = self.conv0(out)

        if self.bn:
            out = out.view(BT, -1, W * self.upsample_factor, H * self.upsample_factor)
            out = self.CBNorm2(out, condition) if self.cbn else self.CBNorm2(out)

        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample_factor != 1:
            out = F.avg_pool2d(out, self.downsample_factor)

        if self.skip_proj:
            skip = x
            if self.upsample_factor != 1:
                skip = F.interpolate(skip, scale_factor=self.upsample_factor)
            skip = self.conv_sc(skip)
            if self.downsample_factor != 1:
                skip = F.avg_pool2d(skip, self.downsample_factor)
        else:
            skip = x

        y = out + skip
        y = y.view(
            BT, -1,
            W * self.upsample_factor // self.downsample_factor,
            H * self.upsample_factor // self.downsample_factor
        )

        return y


if __name__ == "__main__":

    n_class = 96
    batch_size = 4
    n_frames = 20

    gResBlock = GResBlock(3, 100, [3, 3])
    x = torch.rand([batch_size * n_frames, 3, 64, 64])
    condition = torch.rand([batch_size, n_class])
    condition = condition.repeat(n_frames, 1)
    y = gResBlock(x, condition)
    print(gResBlock)
    print(x.size())
    print(y.size())

    # with SummaryWriter(comment='gResBlock') as w:
    #     w.add_graph(gResBlock, [x, condition, ])