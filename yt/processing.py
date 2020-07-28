import torch 
import numpy as np 
import cv2
import pandas as pd


# sets the diagonal (+- thickness/2) to color 
def redx(image, col = [255,0,0], thickness = 3, alpha = .1): 
    col = np.array(col)[:,np.newaxis]
    image = np.copy(image)
    image = (image * alpha).astype(np.uint8)
    S = image.shape[1]
    x = np.concatenate([np.arange(S), np.arange(S)])
    y = np.concatenate([np.arange(S), np.flip(np.arange(S))])
    image[:, x, y] = col
    #print(image.shape, x.shape, y.shape, col.shape, image[:,x,y].shape)
    return image

#samples tenstor and compares consecutive images, returns false, if all samples show constant (difference < eps) imaes
def has_motion(x, n_tries = 8, dist = 5, eps = 1e-3): 
    T, C, H, W = x.shape
    P = H*W
    n_tries = min(n_tries, T)
    step = (T-dist -1) // n_tries
    for i in range(n_tries): 
        i = i*step 
        c = torch.abs(x[i]- x[i+dist]) > eps
        if torch.sum(c) > .1*P: #significant cnage in at least 10% of pixels should be good enough 
            return True
    return False
