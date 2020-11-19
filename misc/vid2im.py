from pandas.core import frame
import torch
import torchvision
import glob
from PIL import Image
import os
import sys

from torchvision.utils import save_image


def cvt(filename, out_dir, step = 1, n_frames = 30): 
    out_dir = os.path.join(out_dir, filename.split(".")[0])
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)

    frames,_,_ = torchvision.io.read_video(filename , pts_unit="sec")
    for i in range(0, min(frames.size(0), n_frames), int(step)): 
       # img = torchvision.transforms.ToPILImage()(frames[i])
       out_file = os.path.join(out_dir, "frame_" + str(i) + ".png")
       save_image(frames[i:i+1].float().permute(0,3,1,2), out_file, normalize=True, padding=0)


if __name__ == '__main__':

    if(len(sys.argv) < 1): 
        print("call this thing with file, (opt step), (opt. tar dir)")
        exit()

    file_dir = sys.argv[1]
    step = sys.argv[2] if len(sys.argv) > 2  else 1
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "/mnt/raid/patrickradner/img_dump/"
    print("in: " + file_dir)
    print("out: " + out_dir)

    if os.path.isdir(file_dir): 
        for file in glob.glob(os.path.join(file_dir, "*.mp4")): 
            d = os.path.join(out_dir, os.path.split(file)[-1].strip(".mp4"))
            print(d)
            os.mkdir(d)
            cvt(file, d, step = step)

    else:         
        cvt(file_dir, out_dir, step = step)
