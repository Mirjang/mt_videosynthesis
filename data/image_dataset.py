import torch
import numpy as np
from torch._C import dtype
import torchvision
import os
from data.base_dataset import BaseDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
from PIL import Image


class ImageDataset(BaseDataset): 


    def initialize(self, opt):
        self.root = opt.dataroot
        #no train/test splits
        if os.path.exists(os.path.join(self.root, opt.phase)): 
            self.root = os.path.join(self.root, opt.phase)

        self.resolution = opt.resolution
        # if opt.phase == "train": #only used for testing atm
        #     self.augmentation = transforms.Compose([
        #     #  torchvision.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
        #     # video_transforms.RandomCrop((self.resolution,self.resolution)),
        #         video_transforms.RandomHorizontalFlip(),
        #         volume_transforms.ClipToTensor(),
        #     ])
        # else: 
        #     self.augmentation = None
        self.use_segmentation = opt.use_segmentation or opt.masked_update
        self.images = sorted(glob.glob(os.path.join(self.root, "gaugan_output*")))
        self.seg = sorted(glob.glob(os.path.join(self.root, "gaugan_input*")))
        self.len = int(min(opt.max_dataset_size, len(self.images)))

        if self.use_segmentation: 
            with open("./data/cocostuff_labels_dynamic.txt") as f: 
                #self.dynamic_indices = torch.tensor([int(x.split(':')[0]) for x in f.read().split('\n')])
                self.dynamic_dict = {int(x.split(':')[0]): x.split(':')[1] for x in f.read().split('\n')}
                self.dynamic_indices = self.dynamic_dict.keys()
              
            with open("./data/cocostuff_labels.txt") as f: 
                self.label_dict = {int(x.split(':')[0]): x.split(':')[1] for x in f.read().split('\n')}


    def __len__(self): 
        return self.len

    def __getitem__(self, index):
        image = transforms.ToTensor()(Image.open(self.images[index]))

        image = F.interpolate(image.unsqueeze(0), size = (self.resolution, self.resolution), mode = "bilinear", align_corners=False).squeeze(0) *255
        image = image.permute(1,2,0) #w,h,c
        out = {'VIDEO': image.unsqueeze(0)}

        if self.use_segmentation: 
            labelmap = np.array(Image.open(self.seg[index]), dtype = np.long)[:,:,1] +1 #indices are in red channel, shifted by 1 
            staticmap = np.zeros_like(labelmap)
            print(labelmap.shape)
            for i in self.dynamic_indices: 
                staticmap[labelmap==i] = 1
            out['SEGMENTATION'] = torch.tensor(staticmap).permute(1,0).unsqueeze(0)
            print(f"{self.images[index]}-{self.seg[index]} found: {[(x, self.label_dict.get(x)) for x in np.unique(labelmap).tolist()]}")
            print(out['SEGMENTATION'].shape)
        return out


