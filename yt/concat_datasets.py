import os
import sys
import shutil
import pandas as pd



def mergeDatasets(src, target, delete_src = True):
    src_meta_file = os.path.join(src, "info.csv")
    target_meta_file = os.path.join(target, "info.csv")
    if not(os.path.exists(src_meta_file) and os.path.exists(target_meta_file)): 
        print("Source or target ds doesnt exist, skipping: "+ src + " -> " + target)
        return
    
    src_df = pd.read_csv(src_meta_file)
    target_df = pd.read_csv(target_meta_file)

    # df = pd.concat([target_df,src_df], keys='video_id')
    # df.drop_duplicates()

    src_df = src_df[src_df.video_id.isin(target_df.video_id) == False]
    src_df.to_csv(target_meta_file, header = False, mode = 'a', index = False)

    for _, row in src_df.iterrows():
        id = row['file_name']
        src2tar = os.path.join(target,id)
        if not os.path.exists(src2tar): 
            if delete_src: 
                shutil.move(os.path.join(src,id), src2tar)  
            else:
                shutil.copy(os.path.join(src,id), src2tar)

    if delete_src: 
        shutil.rmtree(src)

def main(): 
    if len(sys.argv) < 3: 
        print("Need at least 2 datasets! First dataset will be target ")

    target = sys.argv[1]

    for x in sys.argv[2:]: 
        mergeDatasets(x, target)




if __name__ == "__main__": 
    main()
