import os
import sys
import glob







if __name__ == "__main__":

    if(len(sys.argv) < 2) or not os.path.isdir(sys.argv[1]): 
        print("call this thing with path to frames")
        exit()
    dir = sys.argv[1]


    files = glob.glob(os.path.join(dir, "*.jpg"))
    segfiles = glob.glob(os.path.join(dir, "*.png"))

    for i, (f,s) in enumerate(zip(files, segfiles)): 
        os.rename(f, os.path.join(dir, f"frame_{i}.jpg"))
        os.rename(s, os.path.join(dir, f"seg_{i}.png"))

