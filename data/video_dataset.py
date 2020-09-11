import torch
import torchvision
import pandas as pd
import sys
import os
import random
from data.base_dataset import BaseDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvideotransforms import video_transforms, volume_transforms

import torch.hub

def deeplab_inference(model, image, raw_image=None, postprocessor=None):
    with torch.no_grad(): 
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = model(image)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        #probs = probs.cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs.cpu().numpy())
            probs = torch.from_numpy(probs)
        labelmap = torch.argmax(probs, axis=0)

    return logits, probs, labelmap

#expected header in info.csv: video_id,file_name,resolution,fps,start,end
class VideoDataset(BaseDataset): 

   #init when not using cyclegan framework
   # def init(self,root, clips_file ="info.csv",max_clip_length = 10.0, fps = 30, max_size = sys.maxsize, ): 
        #torchvision.set_video_backend("video_reader")

    def initialize(self, opt):
        self.root = opt.dataroot
        #no train/test splits
        clips_file = opt.phase + "_" + opt.clips_file
        if not os.path.exists(os.path.join(self.root, clips_file)):
            clips_file = opt.clips_file

        self.max_clip_length = opt.max_clip_length
        self.fps = opt.fps
        self.skip_frames = opt.skip_frames
        self.df = pd.read_csv(os.path.join(self.root,clips_file))
        self.len = int(min(opt.max_dataset_size, self.df.shape[0]))
        self.nframes = int(opt.fps * opt.max_clip_length // opt.skip_frames)
        self.resolution = opt.resolution
        print(f"nframes: {self.nframes}")
        if opt.phase == "train": 
            self.augmentation = transforms.Compose([
            #  torchvision.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
            # video_transforms.RandomCrop((self.resolution,self.resolution)),
                video_transforms.RandomHorizontalFlip(),
                volume_transforms.ClipToTensor(),
            ])
        else: 
            self.augmentation = None
        self.use_segmentation = opt.use_segmentation
        if self.use_segmentation: 
            self.deeplab = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182).cpu()
            self.deeplab.eval()


    def __len__(self): 
        return self.len

    def __getitem__(self, index):
        clip = self.df.iloc[index]
        start = random.uniform(clip['start'], clip['end'] - self.max_clip_length)
        end = min(start + self.max_clip_length, clip['end'])
        frames, _, info = torchvision.io.read_video(os.path.join(self.root,clip['file_name']), start, end, pts_unit="sec")

        if frames.shape[0] < self.nframes*self.skip_frames: 
            print(f"ERROR: id: {index} has {frames.shape[0]}/{self.nframes*self.skip_frames} frames. File name: {clip['file_name']}")
            missing = self.nframes * self.skip_frames - frames.shape[0]
            frames = torch.cat([frames, frames[-1,...].repeat(missing, 1, 1, 1)])

        if self.skip_frames>1: 
            T,C,H,W = frames.shape
            skipped = torch.zeros(T//self.skip_frames, C,H,W)
            for i in range(T//self.skip_frames):
                skipped[i] = frames[i*self.skip_frames]
            frames = skipped
        frames = frames[:self.nframes,...].float()
        frames = F.interpolate(frames.permute(0,3,1,2), size = (self.resolution, self.resolution), mode = "bilinear", align_corners=False).permute(0,2,3,1)

        if self.augmentation: 
            frames = self.augmentation(frames.numpy()).permute(1,2,3,0) *255
        out = {'VIDEO':frames}
        if self.use_segmentation: 
            first_frame = frames[0].permute(2,0,1).unsqueeze(0)
            logits, probs, labelmap = deeplab_inference(self.deeplab, first_frame)
            out['SEGMENTATION'] = probs

        return out


