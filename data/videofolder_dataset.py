import torch
import torchvision
import pandas as pd
from PIL import Image
import sys
import glob
import os
import random
from data.base_dataset import BaseDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvideotransforms import video_transforms, volume_transforms

import torch.hub
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

#https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (2, 2, 2, 2), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)

def deeplab_inference(model, image, raw_image=None, postprocessor=None):
    with torch.no_grad(): 
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = model(image)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        #probs = probs.cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs.cpu().numpy())
            probs = torch.from_numpy(probs)
        labelmap = torch.argmax(probs, axis=0)

    return logits, probs, labelmap  

def motion_segmentation(x, eps = 1e-3, channel_dim = -1, num_offsets = 3, offset_offsets = 3, smooth_kernel = 5, smooth_sigma = 1): 
    assert len(x.shape) == 4
    seg = 0
    for o in range(1,num_offsets+1):
        o = o * offset_offsets
        td = x - torch.cat([x[:o,...], x[:-o]]) 
        td = torch.abs(td)
        td = torch.sum(td, dim = channel_dim, keepdim = True)
        motion_seg = torch.sum(td, dim = 0) / x.size(0)
        motion_seg = motion_seg > eps
        seg = seg + motion_seg
    seg = seg>0
    if smooth_kernel>0: 
        smoothing = GaussianSmoothing(1, smooth_kernel, smooth_sigma)
        seg = smoothing(seg.permute(2,0,1).float().unsqueeze(0)).squeeze(0).permute(1,2,0)
        seg = seg > 1./smooth_kernel
    return seg.long()


#expected header in info.csv: video_id,file_name,resolution,fps,start,end
class VideofolderDataset(BaseDataset): 

   #init when not using cyclegan framework
   # def init(self,root, clips_file ="info.csv",max_clip_length = 10.0, fps = 30, max_size = sys.maxsize, ): 
        #torchvision.set_video_backend("video_reader")

    def initialize(self, opt):
        self.root = os.path.join(opt.dataroot, opt.phase)

        self.max_clip_length = opt.max_clip_length
        self.fps = opt.fps
        self.skip_frames = opt.skip_frames
        
        dirs = glob.glob(os.path.join(self.root,"*"))
        self.samples = []
        for dir in dirs: 
            self.samples += [os.path.join(dir, sub_dir) for sub_dir in glob.glob(os.path.join(dir,"*"))]

      #  print(self.root, dirs, self.samples)
        self.len = int(min(opt.max_dataset_size, len(self.samples)))
        self.nframes = int(opt.fps * opt.max_clip_length // opt.skip_frames)
        self.resolution = opt.resolution
        print(f"nframes: {self.nframes}")
        if opt.phase == "train" and not opt.no_augmentation: 
            self.augmentation = transforms.Compose([
            #  torchvision.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
            # video_transforms.RandomCrop((self.resolution,self.resolution)),
                video_transforms.RandomHorizontalFlip(),
                volume_transforms.ClipToTensor(),
            ])
        else: 
            self.augmentation = None
        self.use_segmentation = opt.use_segmentation or opt.masked_update
        self.seg_eps = opt.motion_seg_eps

    def __len__(self): 
        return self.len

    def __getitem__(self, index):
        with torch.no_grad(): 
            out = {}
            dir = self.samples[index]
            
            frame_list = [transforms.ToTensor()(Image.open(frame).convert("RGB")) for frame in  glob.glob(os.path.join(dir,"*.png"))]
            frame_list += [transforms.ToTensor()(Image.open(frame).convert("RGB")) for frame in  glob.glob(os.path.join(dir,"*.jpg"))]
    
            frames = torch.stack(frame_list, dim = 0).permute(0,2,3,1)*255

            if frames.shape[0] < self.nframes*self.skip_frames: 
                print(f"ERROR: id: {index} has {frames.shape[0]}/{self.nframes*self.skip_frames} frames. File name: {dir}")
                missing = self.nframes * self.skip_frames - frames.shape[0]
                frames = torch.cat([frames, frames[-1,...].repeat(missing, 1, 1, 1)])

            if self.skip_frames>1: 
                T,C,H,W = frames.shape
                skipped = torch.zeros(T//self.skip_frames, C,H,W)
                for i in range(T//self.skip_frames):
                    skipped[i] = frames[i*self.skip_frames]
                frames = skipped
            first_frame = frames[0]
            frames = frames[:self.nframes,...].float()
            frames = F.interpolate(frames.permute(0,3,1,2), size = (self.resolution, self.resolution), mode = "bilinear", align_corners=False).permute(0,2,3,1)

            if self.augmentation: 
                frames = self.augmentation(frames.numpy()).permute(1,2,3,0) *255
            out['VIDEO'] = frames
            # if self.use_segmentation: 
            #     first_frame = frames[0].permute(2,0,1).unsqueeze(0)
            #     logits, probs, labelmap = deeplab_inference(self.deeplab, first_frame)
            #     staticmap = torch.zeros_like(labelmap)
            #     for i in self.dynamic_indices: 
            #         staticmap[labelmap==i] = 1
            #     out['SEGMENTATION'] = staticmap.unsqueeze(0)
            #     print(f"found: {[(x, self.label_dict[x]) for x in torch.unique(labelmap).tolist()]}")
            #     #out['SEGMENTATION'] = probs
            if self.use_segmentation: 
                out['SEGMENTATION'] = motion_segmentation(frames, eps = self.seg_eps).permute(2,0,1)
            return out


