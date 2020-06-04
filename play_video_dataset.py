import cv2
import os
import sys
from options.train_options import TrainOptions
from data import CreateDataLoader

from data.video_dataset import VideoDataset
from data.movingmnist_dataset import MovingMNISTDataset

if __name__ == '__main__':
    opt = TrainOptions().parse()

    fps = 30
    opt.dataroot = "../datasets/"
    opt.dataset_mode = "movingmnist"

    # opt.dataroot = "../datasets/3hr"
    # opt.dataset_mode = "video"

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    #dataset = MovingMNISTDataset(opt)

    for i, data in enumerate(dataset): 
        clip = data["VIDEO"][0]
        T,_,_,_ = clip.shape
        print(clip.shape)
        for i in range(T):
            frame = clip[i].numpy()#.transpose(1,2,0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("1", frame)
            cv2.waitKey(int(1.0/float(fps)*1000))
    
