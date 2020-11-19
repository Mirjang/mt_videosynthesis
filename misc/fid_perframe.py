from genericpath import exists
import os
import sys
import glob
from shutil import copyfile


if __name__ == "__main__": 
    if(len(sys.argv) < 2): 
        exit()
    rd = sys.argv[1]
    fd = sys.argv[2]
    n_frames = 150

    for i in range(n_frames): 
        #real = os.path.join(rd, str(i))
        fake = os.path.join(fd, str(i))

        os.system(f"python -m pytorch_fid {rd} {fake} >> per_frame_fid.txt")
