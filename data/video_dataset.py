import torch
import torchvision
import pandas as pd

class Clip(): 
    def __init__(self): 
        self.start_time = 0
        self.end_time = 0
        self.length = 0 


class VideoDataset(torch.utils.data.Dataset): 


    def __init__(self, clips_file): 
        self.clips = []

        self.df = pd.read_csv(clips_file, header = None)


        pass


    def __len__(self): 

        return self.df.shape[0]


    def __getitem__(self, index):
        clip = self.df.iloc[index]
        return clip


