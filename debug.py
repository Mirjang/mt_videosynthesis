import time
import os
from options.train_options import TrainOptions
from data import CreateDataLoader, SplitDataLoader
from data.video_dataset import VideoDataset
from models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
from PIL import Image
from sanity_check import sanity_check
import copy

if __name__ == '__main__':
    opt = TrainOptions().parse()
    epsl = [25, 20, 15, 10, 5]

    datas = []
    for eps in epsl:
        opt.motion_seg_eps = eps
        # data_loader = CreateDataLoader(copy.deepcopy(opt))
        # dataset = data_loader.load_data()
        ds = VideoDataset() 
        ds.initialize(copy.deepcopy(opt))
        datas.append(ds)

    visualizer = Visualizer(opt)
    visualizer.reset()

    for i in range(50): 
        segs = []
        for d in datas: 
            batch= d[i]
            img = batch["VIDEO"].unsqueeze(0)[:,0,...].permute(0,3,1,2)
            seg = batch["SEGMENTATION"].unsqueeze(0).expand(-1,3,-1,-1)*255
            segs.append(seg)
     #   print(img.shape, seg.shape)
        images = torch.cat([img] + segs, dim = -1)
        visualizer.vis.images(images, nrow=5, win=i, padding=1, opts=dict(title=f'{i} images'))


        if i> 100: 
            break