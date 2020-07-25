import sys
import numpy as np
import pandas as pd
import os
import visdom 
import torch 
import torchvision
from yt.processing import redx, has_motion

def split_at_cuts(df, root="./"): 

    return df

def remove_constants(df, root="./", print_freq = 10, max_length = 10.0): 
    print("Removing constant vids!")
    out = pd.DataFrame(None, columns = df.columns)
    for it, clip in df.iterrows(): 
        end = min(clip['end'], max_length)
        frames, _, info = torchvision.io.read_video(os.path.join(root,clip['file_name']), clip['start'], end, pts_unit="sec")
        frames = frames.float()/256.0
        if has_motion(frames, n_tries = 8, dist =5, eps = 1e-3): 
            out = out.append(clip, ignore_index = True)
        if it % print_freq == 0: 
            print(f"it: {it} -- out: {out.shape[0]} removed: {it-out.shape[0]+1}")

    print(f"Remove constands Done:\nin: {df.shape[0]} out: {out.shape[0]} removed: {df.shape[0]-out.shape[0]}")
    return out


def manual_edit(vis, df, root="./"):

    return df

def init_manual_interface(env, port = 8197, server = "http://localhost"):

        vis = visdom.Visdom(server=server, port=port, env=env, raise_exceptions=True)





if __name__ == "__main__":

    root = sys.argv[1]

    df = pd.read_csv(os.path.join(root,"info.csv"))

    df = remove_constants(df, root=root)

    vis = init_manual_interface("edit_" + str(root))

