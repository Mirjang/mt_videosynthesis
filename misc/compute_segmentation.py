import torch
import numpy as np
import torch.hub




def compute_segmentation(image, model = None): 
    if model is None: 
        model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182)

    