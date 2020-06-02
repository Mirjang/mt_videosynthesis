import torch
import torchvision
import pandas as pd
import sys
import os
from data.base_dataset import BaseDataset

class Clip(): 
    def __init__(self): 
        self.start_time = 0
        self.end_time = 0
        self.length = 0 

#expected header in info.csv: video_id,file_name,resolution,fps,start,end
class VideoDataset(BaseDataset): 

   #init when not using cyclegan framework
   # def init(self,root, clips_file ="info.csv",max_clip_length = 10.0, fps = 30, max_size = sys.maxsize, ): 
        #torchvision.set_video_backend("video_reader")

    def initialize(self, opt):
        self.root = opt.dataroot   
        self.max_clip_length = opt.max_clip_length
        self.fps = opt.fps

        self.df = pd.read_csv(os.path.join(root,opt.clips_file))
        self.len = int(min(opt.max_dataset_size, self.df.shape[0]))

    def __len__(self): 
        return self.len

    def __getitem__(self, index):
        clip = self.df.iloc[index]
        start = clip['start']
        end = min(clip['end'], start + self.max_clip_length)
        frames, _, info = torchvision.io.read_video(os.path.join(self.root,clip['file_name']), start, end, pts_unit="sec")

        return frames


