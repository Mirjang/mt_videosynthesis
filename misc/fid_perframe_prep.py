from genericpath import exists
import os
import sys
import glob
from shutil import copyfile


if __name__ == "__main__": 
    if(len(sys.argv) < 2): 
        exit()
    indir = sys.argv[1]
    odir = sys.argv[2]
    n_frames = 30
    os.makedirs(odir, exist_ok=True)

    for i in range(n_frames): 
        ipath = os.path.join(indir, f"*_{i}.png" )
        opath = os.path.join(odir, str(i))
        os.mkdir(opath)
        os.system(f"cp {ipath} {opath}")
