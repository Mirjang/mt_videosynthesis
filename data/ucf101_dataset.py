import torch
import torchvision
import torch.nn.functional as F
import pandas as pd
import sys
import os
import random
from data.base_dataset import BaseDataset
import glob

import subprocess

#https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def parse_dataset(root, clips_file = "info.csv", clip_length = 1.0, skip_length = -1.0, write = True, safe = True, min_nframes = 30):
    i = 0
    skip_length = max(clip_length, skip_length)
    df = pd.DataFrame(None, columns = ["file_name", "start", "end"])
    vids = glob.glob(os.path.join(root, "*.mp4")) + glob.glob(os.path.join(root, "*.avi"))
    for file_name in vids:
        vid_length = get_length(file_name) - 0.1 #just to make sure we dont get a shorter clip
        for l in range(int(vid_length/skip_length)): 
            start = l*skip_length
            end = start + skip_length
            if safe: 
                frames, _, info = torchvision.io.read_video(os.path.join(root, file_name), start, end, pts_unit="sec")
                if frames.shape[0] < min_nframes: 
                    continue
                
            df = df.append({"file_name":file_name, "start": start, "end": end}, ignore_index=True)

        if i % 100 == 0:
            print(f"video parsing: {i}/{len(vids)} - Num Clips: {df.shape[0]}")   
        i +=1 

    print(f"Done parsing {i} videos into {df.shape[0]} clips.")
    if write: 
        df.to_csv(os.path.join(root, clips_file), header = True, mode = 'w', index = False)
    return df

#expected header in info.csv: video_id,file_name,resolution,fps,start,end
class UCF101Dataset(BaseDataset): 

   #init when not using cyclegan framework
   # def init(self,root, clips_file ="info.csv",max_clip_length = 10.0, fps = 30, max_size = sys.maxsize, ): 
        #torchvision.set_video_backend("video_reader")

    def initialize(self, opt):
        self.root = os.path.join(opt.dataroot, opt.phase)
        #no train/test splits
        if not os.path.exists(os.path.join(opt.dataroot, opt.phase)):
            self.root = opt.dataroot
        self.max_clip_length = opt.max_clip_length
        self.fps = opt.fps
        self.skip_frames = opt.skip_frames
        self.nframes = int(opt.fps * opt.max_clip_length // opt.skip_frames)

        clip_file_abs = os.path.join(self.root, opt.clips_file)
        if not os.path.exists(clip_file_abs) and len(glob.glob(os.path.join(self.root, "*.mp4")) + glob.glob(os.path.join(self.root, "*.avi")))>0 or opt.reparse_data: 
            actual_length = self.max_clip_length * self.skip_frames
            skip_length = actual_length * 2 # dont use back2back frames to save compute
            self.df = parse_dataset(self.root, clips_file = opt.clips_file, clip_length = actual_length, skip_length = skip_length, write=True, safe = True, min_nframes = self.nframes)
        else: 
            self.df = pd.read_csv(clip_file_abs)
        self.len = int(min(opt.max_dataset_size, self.df.shape[0]))
        self.resolution = opt.resolution
    def __len__(self): 
        return self.len

    def __getitem__(self, index):
        clip = self.df.iloc[index]
        # start = random.uniform(clip['start'], clip['end'] - self.max_clip_length)
        # end = min(start + self.max_clip_length, clip['end'])
        start = clip["start"]
        end = min(clip["end"], start + self.max_clip_length * self.skip_frames)
        frames, _, info = torchvision.io.read_video(os.path.join(self.root,clip['file_name']), start, end, pts_unit="sec")
        if self.skip_frames>1: 
            T,C,H,W = frames.shape
            skipped = torch.zeros(T//self.skip_frames, C,H,W)
            for i in range(T//self.skip_frames):
                skipped[i] = frames[i*self.skip_frames]
            frames = skipped

        if frames.shape[0] < self.nframes: 
            print(f"ERROR: id: {index} has {frames.shape[0]}/{self.nframes} frames. File name: {clip['file_name']}")
            missing = self.nframes - frames.shape[0]
            frames = torch.cat([frames, frames[-1,...].repeat(missing, 1, 1, 1)])
        frames = F.interpolate(frames[:self.nframes,...].float().permute(0,3,1,2), size = (self.resolution, self.resolution), mode = "bilinear", align_corners=False).permute(0,2,3,1)

        return {'VIDEO':frames[:self.nframes]}


