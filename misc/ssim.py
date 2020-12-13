#src: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

import math
import numpy as np
import cv2
import glob

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def load_vids(dir, resolution = 224,n_frames = 24, file_type ="*.mp4", B = 200): 
  np_tensor = None
  it = 0
  for file in glob.glob(os.path.join(dir, file_type)): 
    it += 1
  
    frames, _, info = torchvision.io.read_video(file, pts_unit="sec")
    frames = F.interpolate(frames.permute(0,3,1,2).float(), size = (resolution+1, resolution+1), mode = "bilinear", align_corners=False)
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
    B = 255
    b = load_vids(dir2, B = B)
    print(f"done loading B {b.shape}")


    a = load_vids(dir1, B = B)
    print(f"done loading A {a.shape}")

    B = min(a.shape[0], b.shape[0])  
    nframes = min(a.shape[1], b.shape[1])  
    print(f"nframes: {nframes}")


    ssims_tot = []
    max_tot = []
    ssim_vid = []
    ssim_vid_max = []

    for ia in range(B): 
        ssims = []
        ssim_vid_i = []
        for ib in range(B): 
            pi_ssim = []
            for f in range(nframes): 
                val = calculate_ssim(a[ia,f], b[ib,f])
                pi_ssim.append(val)
                ssims.append(val)
            ssim_vid_i.append(np.mean(pi_ssim))
        ssim_vid.append(ssim_vid_i.copy())
        ssim_vid_max.append(np.max(ssim_vid_i))

        ssims_tot += ssims.copy()
        max_tot.append(np.max(ssims))
        print(f"image {ia}: min: {np.min(ssims)}, max: {np.max(ssims)}, mean: {np.mean(ssims)}")
    
    print(f"\ntotal: min: {np.min(ssims_tot)}, max: {np.max(ssims_tot)}, mean: {np.mean(ssims_tot)}, avg_max: {np.mean(max_tot)}, max_agv: {np.mean(ssim_vid)}, max_max: {np.mean(ssim_vid_max)}")
