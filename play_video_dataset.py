import cv2
import os
import sys

from data.video_dataset import VideoDataset

if __name__ == '__main__':
    
    fps = 30
    dataset = VideoDataset("../datasets/Relaxing_3_Hour_Video_of_a", max_size =1, max_clip_length = 60.0)

    for i in range(len(dataset)): 
        clip = dataset[i]
        T,_,_,_ = clip.shape
        for i in range(T):
            frame = clip[i].numpy()#.transpose(1,2,0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("1", frame)
            cv2.waitKey(int(1.0/float(fps)*1000))
    
