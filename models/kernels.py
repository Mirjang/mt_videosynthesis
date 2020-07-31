import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class ExponentialKernel(): 
    def __init__(self, sigma = 1.):
        super().__init__()
        self.twosigma = sigma * 2

    def __call__(self, x, scale = 1.): 
        return scale * torch.exp(-x/(self.twosigma))



class TruncatedExponentialKernel(): 
    def __init__(self, sigma = 1., truncation_dist = .8):
        super().__init__()
        self.twosigma = sigma * 2
        self.truncation_dist = truncation_dist

    def __call__(self, x, scale = 1.): 
        x = torch.exp(-x/(self.twosigma))
        x = torch.clamp_min(x, self.truncation_dist)
        return scale * x


