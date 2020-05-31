import os
import sys
import numpy as np
from PIL import Image
from data.transparent_dataset import loadRGBAFloatEXR


if __name__ == '__main__':

    if(len(sys.argv) < 3): 
        print("call this thing with src and tar dir")
        exit()

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    print("in: " + in_dir)
    print("out: " + out_dir)
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)

    i = 0 
    while os.path.exists(os.path.join(in_dir, str(i) + "_rgb.exr")):
        rgb = os.path.join(in_dir, str(i) + "_rgb.exr")
        if i % 25 == 0:
            print("Processing Image: " + rgb)
        rgb_array = loadRGBAFloatEXR(rgb, ['R', 'G', 'B'])
        im = Image.fromarray((rgb_array*255).astype(np.uint8))
        im.save(os.path.join(out_dir, str(i) + "_rgb.png"))
        i = i+1

    print("DONE: ")
