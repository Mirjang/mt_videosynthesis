# src: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision import transforms

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import glob


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception modela
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == "__main__":
    import sys
    import os
    import torchvision
    import pandas as pd
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    if len(sys.argv) < 1: 
        print(f"SYNTAX: {sys.argv[0]} img-dir")
        sys.exit(-1)

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)
    class ProperNorm(object): 
        def __call__(self,sample): 
            if sample.shape[-1]< 128: #is requites at least 128 res
                
                sample = F.interpolate(sample.unsqueeze(0),size = (128,128)).squeeze()
            return sample * 2.0 - 1.0
    t = transforms.Compose([transforms.ToTensor(), ProperNorm()])
    imgs = torchvision.datasets.ImageFolder(sys.argv[1], transform=t)
    imgs = IgnoreLabelDataset(imgs)

    print(f"done loading A {imgs} - {len(imgs)}")
    score = inception_score(imgs, cuda = False)
    print(f"IS is: {score}")