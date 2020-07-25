import torch 
import numpy as np 

# sets the diagonal (+- thickness/2) to color 
def redx(tensor, col = [1.,0.,0.], thickness = 3): 

    T = tensor.shape[0]
    W = tensor.shape[-2]
    H = tensor.shape[-1]
    tensor *= .1 #darken everything

    mask = torch.eye(W,H).byte().expand(T, -1, -1)
    print(mask.shape)
    tensor = tensor.permute(1,0,2,3)
    print(tensor[:,mask].shape)
    tensor[:, mask] = torch.tensor(col).unsqueeze(1).float()
    tensor = tensor.permute(1,0,2,3)
    
    return tensor

#samples tenstor and compares consecutive images, returns false, if all samples show constant (difference < eps) imaes
def has_motion(x, n_tries = 10, dist = 5, eps = 1e-3): 
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

