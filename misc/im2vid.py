from pandas.core import frame
import torch
import torchvision
import glob
from PIL import Image
import os
import sys

from torchvision.utils import save_image


def cvt(root, nframes = 25): 

    for dir in glob.glob(root + "/*/"): 
        frame_list = [torchvision.transforms.ToTensor()(Image.open(frame).convert("RGB")) for frame in sorted(glob.glob(os.path.join(dir, "*.png")))]
        if len(frame_list) < 1: 
            continue

        frames = torch.stack(frame_list, dim = 0).permute(0,2,3,1)*255
        out_name = dir.strip("/") + ".mp4"
        torchvision.io.write_video(out_name, frames, 25/2)


if __name__ == '__main__':

    if(len(sys.argv) < 1): 
        print("call this thing with root dir argument")
        exit()

    root = sys.argv[1]
    cvt(root)
