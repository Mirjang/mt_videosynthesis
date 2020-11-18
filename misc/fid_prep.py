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

    os.makedirs(odir, exist_ok=True)


    for root, dirnames, filenames in os.walk(indir): 
        for filename in filenames: 
            if filename.endswith(".png"): 
                ipath = os.path.join(root, filename)
                oname = ipath.replace("/", "_").strip("._")
                copyfile(ipath, os.path.join(odir, oname))