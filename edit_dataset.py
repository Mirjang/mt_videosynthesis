import sys
import numpy as np
import pandas as pd
import math
import os
import visdom 
import torch 
import torch.nn.functional as F
import torchvision
from yt.processing import redx, has_motion
from util.util import tensor2im
from util.visualizer import imresize
from time import sleep
import copy

def insert_file_suffix(name, suffix): 
    parts = name.split(".")
    parts[0] += str(suffix)
    return ".".join(parts)


def subsample_raw(df, root="./",
                fps = 30, 
                clip_length = 10.0,
                crop_frist_last = 20.0,
                max_samples_per_vid = 25, 
                min_distance = 5.0, 
                print_freq = 10,
                remove_constants = True, 
                delete_original = False,
                write_video = True): 
    print("Subsampling raw videos!")
    out = pd.DataFrame(None, columns = df.columns)

    for it, raw_vid in df.iterrows: 
        file_name = os.path.join(root,raw_vid['file_name'])
        start = clip['start'] #this should always be 0 at this point... i think 
        end = clip['end']
        raw_length = end - start
        if raw_length - 2 * crop_frist_last > clip_length: #unless we are given some very short vid, we just discard possible intros/outros
            start = start + crop_frist_last
            end = end - crop_frist_last
            raw_length = end - start

        step = max(clip_length+min_distance, raw_length/max_samples_per_vid) #largest possible temporal distance between samples
        good_clips = 0
        for i in range(max_samples_per_vid): 
            clip_start = start + i* step
            if clip_start + step > end: 
                break

            frames, _, info = torchvision.io.read_video(file_name, clip_start, clip_start+clip_length, pts_unit="sec")
            if not remove_constants or has_motion(frames, n_tries=8, dist = 5, eps = 1e-3): 
                entry = copy.copy(raw_vid)
                clip_name = insert_file_suffix(clip['file_name'], f"_clip{good_clips}")
                entry['file_name'] = clip_name
                entry['start']=0
                entry['end']=clip_length
                if write_video: #disable mainly for debugging
                    torchvision.io.write_video(os.path.join(root,clip_name), frames, fps = info["video_fps"])
                out = out.append(entry)
                good_clips += 1

        if delete_original: 
            os.remove(file_name)

        if it % print_freq == 0: 
            print(f"it: {it} -- out: {out.shape[0]} removed: {it-out.shape[0]+1}")

    print(f"Subsample raw Done:\nin: {df.shape[0]} out: {out.shape[0]}")

    return df


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



class ManualEdit(): 

    def __init__(self, env, port = 8197, server = "http://localhost", root = "./"):
        self.vid_panel = '1' #REALLY importatnt that this is a string!!
        self.thumbnail_panel = '2'
        self.control_panel = '3'
        self.root = root
        self.next = True
        self.abort = False
        self.vis = visdom.Visdom(server=server, port=port, env=env, raise_exceptions=True)
        self.enable_input = False

        button_ok = torch.zeros((3,128,128))
        button_ok[1,...] = 1
        button_cancel = torch.zeros((3,128,128))
        button_cancel[0,...] = 1
        grid = torch.cat([button_cancel, button_ok], dim = 1)
        self.vis.image(grid,win = self.control_panel, opts=dict(title="Control Panel"))
        self.vis.register_event_handler(self, self.control_panel)
        self.vis.register_event_handler(self, self.vid_panel)
        self.vis.register_event_handler(self, self.thumbnail_panel)

    def __call__(self,event): 
        if not self.enable_input: 
            return
        target = event['target']
        pane_data = event['pane_data']
     #   print(target)
        #window specific events
        if target == self.control_panel: 
            if event['event_type'] == 'Click': 
                x,y = event['image_coord']['x'], event['image_coord']['y']
                print("ctrl:", x,y)
        # interaction w/ vid doesnt work
        # elif target == self.vid_panel: 
        #     if event['event_type'] == 'Click': 
        #         x,y = event['image_coord']['x'], event['image_coord']['y']
        #         print("vid: ", x,y)
        elif target == self.thumbnail_panel: 
            if event['event_type'] == 'Click': 
                x,y = event['image_coord']['x'], event['image_coord']['y']
                ix = x // self.thumbnail_res
                iy = y // self.thumbnail_res
                idx = ix + iy * self.nrows
                print("thumb: ", x,y, "-->", ix, iy, idx)
                if ix < self.ncols and iy < self.nrows: 
                    selected = self.selected[idx]
                    self.selected[idx] = not selected

                    if selected: #deselect
                        self.thumbnail_display[idx] = redx(self.thumbnails[idx])
                    else: #reselect
                        self.thumbnail_display[idx] = self.thumbnails[idx]
                # self.vis.image(self.thumbnail, win = self.thumbnail_panel,opts=dict(title="thumbnails (select here)"))
                    self.vis.images(self.thumbnail_display, win = self.thumbnail_panel, nrow = self.nrows,opts=dict(title="thumbnails (select here)"))


        #global events
        if event['event_type'] == 'KeyPress':
            if event['key'] == 'Enter':
                print("next batch")
                self.next = True
            elif event['key'] == 'Escape': 
                print("Abort()")     
                self.abort = True
            

    def run(self, df, max_length = 1.0, fps = 30, nrows = 5, ncols = 5, res = 64, thumbnail_res = 128, tick_rate = .05):
        n_elems = nrows * ncols
        n_vids = df.shape[0]
        B = (n_vids+1)//n_elems
        batch = 0       
        self.nrows = nrows
        self.ncols = ncols
        self.thumbnail_res = thumbnail_res

        out = pd.DataFrame(None, columns = df.columns)
        for b in range(B):   
            self.next = False
            batch = df[b*n_elems: min(n_vids, (b+1)*n_elems)]

            self.display_batch(df,batch, max_length = max_length, fps = fps, nrows = nrows, ncols = ncols, res = res, thumbnail_res = thumbnail_res)
            self.selected = [True] * batch.shape[0] #rows
            print(f"Manual batch: {b+1}/{B}")

            self.enable_input=True
            while not self.next: # wait for user to do their thing   
                if self.abort: 
                    return None
                sleep(tick_rate)
            self.enable_input=False    
            
            # collect info on selected vids and store output df
            good_ones = batch[self.selected]
            out = out.append(good_ones)
            print(f"Selected {good_ones.shape[0]}/{batch.shape[0]} clips. Total: {out.shape[0]}")

        return out

    def display_batch(self, df,batch, max_length = 5.0, fps = 30, nrows = 4, ncols = 3, res = 32, thumbnail_res = 32, aspect_ratio = 1.0):
        n_frames = int(max_length * fps)
   
        display = torch.ones([n_frames, 3, res *ncols, res*nrows])
        i = -1 #iterrows returns index, not enumerator
        thumbnails = []
        for _, clip in batch.iterrows(): 
            i += 1
            end = min(clip['end'], max_length)
            frames, _, info = torchvision.io.read_video(os.path.join(self.root,clip['file_name']), clip['start'], end, pts_unit="sec")
            frames = frames.float()/256.0
            thumbnail = F.interpolate(frames[:1].permute(0,3,1,2), size = (int(thumbnail_res*aspect_ratio) , thumbnail_res), mode ="bilinear", align_corners=False)
            thumbnail = tensor2im(thumbnail).astype(np.uint8).transpose(2,0,1)
            thumbnails.append(thumbnail)
            frames = F.interpolate(frames[:n_frames,...].float().permute(0,3,1,2), size = (res, res), mode = "bilinear", align_corners=False)
            
            if frames.shape[0] < n_frames: 
                missing = n_frames - frames.shape[0]
                frames = torch.cat([frames, frames[-1,...].repeat(missing, 1, 1, 1)])

            y = i % nrows * res
            x = i // nrows * res

            display[:, :, x:x+res, y:y+res] = frames

        self.thumbnails = thumbnails #stores actual images
        self.thumbnail_display = thumbnails.copy() # stores displayed images (alterd if deselected)
        self.vis.images(self.thumbnails, win = self.thumbnail_panel, nrow = self.nrows,opts=dict(title="thumbnails (select here)"))

        #self.vis.image(self.thumbnail, win = self.thumbnail_panel,opts=dict(title="thumbnails (select here)"))

        self.vis.video(display.permute(0,2,3,1)[...,[2,1,0]], win = self.vid_panel,opts=dict(title="videos",fps=fps))
            
if __name__ == "__main__":

    root = sys.argv[1]

    df = pd.read_csv(os.path.join(root,"info.csv"))

 #   df = remove_constants(df, root=root)
    df = subsample_raw(df, root=root, write_video=False)
#    manual_interface = ManualEdit("edit",root = root)
 #   df = manual_interface.run(df)

