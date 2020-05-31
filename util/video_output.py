import numpy as np
import os
import sys
import ntpath
import time
from . import util
from . import html
from scipy.misc import imresize
#from PIL import Image


from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import cv2


class VideoOutput(): 

    def __init__(self, opt): 
        self.out_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
        self.out_file = os.path.join(self.out_dir, opt.name + opt.phase +  ".avi")
        self.writer = None

        if(os.path.exists(self.out_file)): 
            os.remove(self.out_file)

    def writeFrame(self, visuals): 

        i = 0
        for label, im_data in visuals.items():
            im = util.tensor2im(im_data)#.transpose(2,1,0)

            if(i==0): 
                display = im
            else: 
                display = np.concatenate((display, im), axis = 1)
            i+=1

        #print(display.shape)
        if(self.writer == None): 
            dims = display.shape[1], display.shape[0]
            self.writer = VideoWriter(self.out_file, VideoWriter_fourcc(*"XVID"), 30.0, dims, True)

        self.writer.write(cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

    def close(self): 
        self.writer.release()