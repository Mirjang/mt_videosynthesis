import torch
import torchvision
import sys
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
        self.use_segmentation = opt.use_segmentation
        self.images = glob.glob(os.path.join(self.root, "frame_*"))
        self.seg = glob.glob(os.path.join(self.root, "seg_*"))
        self.len = int(min(opt.max_dataset_size, len(self.images)))

    def __len__(self): 
        return self.len

    def __getitem__(self, index):
        image = transforms.ToTensor()(Image.open(self.images[index]))

        image = F.interpolate(image.unsqueeze(0), size = (self.resolution, self.resolution), mode = "bilinear", align_corners=False).squeeze(0) *255
        image = image.permute(1,2,0) #w,h,c
        out = {'VIDEO': image.unsqueeze(0)}
        # if self.use_segmentation: 
        #     out['SEGMENTATION'] = probs

        return out


