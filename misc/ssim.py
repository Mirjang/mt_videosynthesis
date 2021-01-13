#src: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

import math
import numpy as np
import cv2
import glob
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def load_vids(dir, resolution = 224,n_frames = 24, file_type ="*.mp4", B = 200): 
  np_tensor = None
  it = 0
  for file in glob.glob(os.path.join(dir, file_type)): 
    it += 1
  
    frames, _, info = torchvision.io.read_video(file, pts_unit="sec")
    frames = F.interpolate(frames.permute(0,3,1,2).float(), size = (resolution, resolution), mode = "bilinear", align_corners=False)
    frames = frames[1:n_frames+1,...].permute(0,2,3,1).unsqueeze(0)
    #print(frames.min(), frames.max())
    if frames.size(1) < n_frames: 
        continue
    if np_tensor is None:
        np_tensor = frames.numpy()
    else: 
        np_tensor = np.concatenate((np_tensor, frames.numpy()), axis = 0)  
   # print(np_tensor.shape)

    if it % 50 == 0: 
        print(f"loading file {file}")
    if np_tensor.shape[0] == B: 
        break
  return np_tensor

if __name__ == "__main__":
    import sys
    import os
    import torchvision
    import pandas as pd
    import numpy as np
    import torch.nn.functional as F
    import copy

    if len(sys.argv) < 2: 
        print(f"SYNTAX: {sys.argv[0]} dir1 dir2")
        sys.exit(-1)
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    B=8
    B = 256
    b = load_vids(dir2, B = B, resolution= 64)
    print(f"done loading B {b.shape}")


    a = load_vids(dir1, B = B, resolution= 64)
    print(f"done loading A {a.shape}")

    # a = torch.rand((256,24,64,64,3))
    # b = torch.rand((256,24,64,64,3))

    B = min(a.shape[0], b.shape[0])  
    nframes = min(a.shape[1], b.shape[1])  
    print(f"nframes: {nframes}")

    a = torch.tensor(a).permute(0,1,4,2,3).float().cuda()/255
    b = torch.tensor(b).permute(0,1,4,2,3).float().cuda()/255
   # print(a.min(), a.max())
    with torch.no_grad(): 
        ssim_loss = SSIM(window_size = 11)
        ssim_tot = []
        per_frame = [[] for _ in range(nframes)]
        for ib in range(B): 
            ssim_per_real = []
            for ia in range(B): 
                ssim_per_frame = []
                for f in range(nframes): 
                #    print(a[ia,f].unsqueeze(0).shape)
                    val = ssim_loss(a[ia,f].unsqueeze(0), b[ib,f].unsqueeze(0)).item()
                    ssim_per_frame.append(val)
                    per_frame[f].append(val)
                ssim_per_real.append(np.mean(ssim_per_frame))


            ssim_tot.append(np.max(ssim_per_real))
            print(ib)
          #  print(f"image {ib}: min: {np.min(ssims)}, max: {np.max(ssims)}, mean: {np.mean(ssims)}")
        per_frame = [np.mean(x) for x in per_frame]
        print(f"\ntotal: ssim: {np.mean(ssim_tot)}, min: {np.min(ssim_tot)}, max: {np.max(ssim_tot)},")
        print([f"{x:1.3f}" for x in per_frame])
