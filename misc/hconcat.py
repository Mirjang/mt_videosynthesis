from sys import argv
import torch
import torchvision
import os
import sys
import glob


if __name__ == "__main__": 
    root = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv)>2 else 1

    files = [root]
    for d in range(depth-1): 
        files_ = []
        for dir in files: 
            files_ += glob.glob(dir)
        files = files_ 
    
    files_ = []
    for dir in files: 
        files_ += glob.glob(os.path.join(dir, "*.mp4"))
    files = files_


    clips = []
    info = {}
    for file in files: 

        clip,_,info = torchvision.io.read_video(file , pts_unit="sec")
        print(clip.shape)
        clips.append(clip)

    cat = torch.cat(clips, dim = 2)
    omp4 = os.path.join(root, "out.mp4")
    owebm = os.path.join(root, "out.webm")
    print(cat.shape)
    torchvision.io.write_video(omp4, cat, info["video_fps"])

    os.system(f"ffmpeg -i {omp4} -c:v libvpx-vp9 -crf 30 -b:v 0 -b:a 128k -c:a libopus {owebm}")