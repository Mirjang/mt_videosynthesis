import sys
import os
import numpy as np
import math
import pandas as pd
import random

def file_in_dir(file, name): 
    path, filename = os.path.split(file)
    return os.path.join(path, name)

def main(file, num_val = 500, write = True): 
    df = pd.read_csv(file)
    df_val = pd.DataFrame(None, columns = df.columns)
    n = df.shape[0]

    while df_val.shape[0] < num_val: 
        i = random.randint(0, df.shape[0]-1)
        video_id = df.iloc[i]["video_id"]
        df_val = df_val.append(df[df["video_id"]==video_id])
        df = df[df["video_id"]!=video_id]

    if write: 
        df.to_csv(file_in_dir(file, "train_info.csv"),header=True, mode = "w", index = False)
        df_val.to_csv(file_in_dir(file, "val_info.csv"),header=True, mode = "w", index = False)

    else: 
        print(df)
        print("-------------val---------------")
        print(df_val)

    print(f"{n} samples split into {df.shape[0]} train and {df_val.shape[0]} validation samples")
    return df, df_val


if __name__ == "__main__": 
    if(len(sys.argv) < 2): 
        print(f"SYNTAX: {sys.argv[0]} path/to/dataset [num_val_clips]")
        exit()
    file = sys.argv[1]
    if os.path.isdir(sys.argv[1]): 
        file = os.path.join(sys.argv[1], "info.csv")

    num_val = 200
    if len(sys.argv) == 3: 
        num_val = int(sys.argv[2])
    main(file, num_val)