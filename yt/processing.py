import os
import sys
import torch 
import numpy as np 
import cv2
import pandas as pd
import subprocess
import glob

# sets the diagonal (+- thickness/2) to color 
def redx(image, col = [255,0,0], thickness = 3, alpha = .1): 
    col = np.array(col)[:,np.newaxis]
    image = np.copy(image)
    image = (image * alpha).astype(np.uint8)
    S = image.shape[1]
    x = np.concatenate([np.arange(S), np.arange(S)])
    y = np.concatenate([np.arange(S), np.flip(np.arange(S))])
    image[:, x, y] = col
    #print(image.shape, x.shape, y.shape, col.shape, image[:,x,y].shape)
    return image

#samples tenstor and compares consecutive images, returns false, if all samples show constant (difference < eps) images
def has_motion(x, n_tries = 8, dist = 5, eps = 1e-3): 
    T, C, H, W = x.shape
    P = H*W
    n_tries = min(n_tries, T)
    step = (T-dist -1) // n_tries

    for i in range(n_tries): 
        i = i*step 
        if i + dist >= T: # vid is too short 
            return False 
        c = torch.abs(x[i]- x[i+dist]) > eps
        if torch.sum(c) > .1*P: #significant change in at least 10% of pixels should be good enough 
            return True
    return False


#https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_video_id(file_name): 
    no_path = os.path.split(file_name)[-1]
    return str(no_path).split(".")[0].split("_clip")[0]


def parse_dataset(root, clips_file = "info.csv", clip_length = 1.0, write = True, safe = True, min_nframes = 30, res = "360p"):
    i = 0
    df = pd.DataFrame(None, columns = ["video_id","file_name","resolution", "fps", "start", "end"])
    vids = glob.glob(os.path.join(root, "*.mp4")) + glob.glob(os.path.join(root, "*.avi"))
    start = 0
    discarded = 0
    for file_name in vids:
        vid_length = get_length(file_name)
        end = vid_length
        use_clip = True
        if safe: 
            frames, _, info = torchvision.io.read_video(os.path.join(root, file_name), start, end, pts_unit="sec")
            if frames.shape[0] < min_nframes or vid_length < clip_length: 
                print(f"Discarding: {file_name} - Frames: {frames.shape[0]}/{min_nframes} - Length: {vid_length}/{clip_length}")
                use_clip = False
                discarded += 1

        if use_clip:
            file_name = os.path.base_name(file_name)
            df = df.append({"video_id":get_video_id(file_name), "file_name":file_name,"resolution": res, "fps": 30, "start": start, "end": end}, ignore_index=True)

        if i % 100 == 0:
            print(f"video parsing: {i+1}/{len(vids)} - Num Clips: {df.shape[0]}")   
        i +=1 

    print(f"Done parsing {i} videos into {df.shape[0]} clips. Discarded: {discarded} clips")
    if write: 
        df.to_csv(os.path.join(root, clips_file), header = True, mode = 'w', index = False)
    return df


if __name__ == "__main__":

    root = sys.argv[1]
    df = parse_dataset(root, clips_file="info_parse.csv", safe=False )
    df.to_csv(os.path.join(root, "info_processed.csv"), header=True, mode = "w", index = False)

