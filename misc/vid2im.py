from pandas.core import frame
import torch
import torchvision
import glob
from PIL import Image
import os
import sys






def cvt(filename, root, out_dir): 
    out_dir = os.path.join(out_dir, filename.split(".")[0])
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)

    frames,_,_ = torchvision.io.read_video(os.path.join(os.path.join(root,filename)) , pts_unit="sec")

    for i in range(frames.size(0)): 
        img = torchvision.transforms.ToPILImage()(frames[i])
        img.save(os.path.join(out_dir, filename.split(".")[0]+ "frame_" + str(i) + ".png"))



if __name__ == '__main__':

    if(len(sys.argv) < 1): 
        print("call this thing with file and (opt. tar dir)")
        exit()

    file_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/raid/patrickradner/img_dump/"
    print("in: " + file_dir)
    print("out: " + out_dir)

    head_tail = os.path.split(file_dir) 

    cvt(head_tail[1], head_tail[0], out_dir)
