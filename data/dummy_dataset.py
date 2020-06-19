import torch
import torchvision
import numpy as np
import sys
import os
import random
from data.base_dataset import BaseDataset

class DummyDataset(BaseDataset): 

    def initialize(self, opt):
        train = opt.phase is "train"
        self.nframes = int(opt.max_clip_length * opt.fps)
        self.cifar10 = torchvision.datasets.CIFAR10(opt.dataroot, train = train, download=True, transform = torchvision.transforms.ToTensor())
        
        self.len = int(min(opt.max_dataset_size, len(self.cifar10)))

        ls = torch.linspace(0,1,steps=self.len)
        self.colors = torch.stack([ls,ls,ls])
        self.size = 6
        self.res = 32
        self.start_pos = self.size//2+1
        self.end_pos = self.res - (self.size//2+1)


        self.background = torch.rand((self.res, self.res,3))*256


    def __len__(self): 
        return self.len

    def __getitem__(self, index):
        #r,g,b = 0,0,255
        r,g,b = random.random() * 255, random.random() * 255, random.random() * 255

        background,_ = self.cifar10[index]
        background = background.permute(1,2,0) *255
        #frames = torch.zeros((self.nframes, self.res, self.res,3))
        frames = background.repeat((self.nframes,1,1,1))
        d = (self.end_pos - self.start_pos) / self.nframes

        box = torch.ones(self.size, self.size)
        box = torch.stack([box*r, box*g, box*b], axis=-1).unsqueeze(0)
        s2 = self.size/2

        #y = int(self.res // 2 - s2)
        y = int(index % (self.res-2*self.size) + self.size )

        for t in range(self.nframes): 
            
            x = int(self.start_pos + int(t*d) - s2)
            frames[t:t+1,y:y+self.size, x:x+self.size,:] = box

        return {'VIDEO':frames}


