import numpy as np
import os
import sys
import shutil
import tempfile
import concurrent.futures
import threading
from pytube import YouTube as Downloader
from youtube_api import YouTubeDataAPI
import pandas as pd
import glob

api = YouTubeDataAPI("YOUR KEY HERE")

resolutions = ['1080p', '720p', '480p', '360p', '240p', '144p']


def download_video(video, path, retries = 5, min_res="240p", max_res="720p"): 
    url = "youtube.com/watch?v=" + video['video_id']
    filename = video['video_id']

    for i in range(retries):
       try: 
        print("Downloading: " + url)
        dl = Downloader(url)
        streams = dl.streams.filter(only_video=True)
        stream = None
        for res in resolutions[resolutions.index(max_res) : resolutions.index(min_res)]: 
            res_streams = streams.filter(resolution = res)
            #print(res_streams)
            if len(res_streams)>0: #at this point we could pick our favourite codec or w/e
                stream = res_streams[0]
                break 
        if not stream: 
            return None
        stream.download(path, filename = filename)
        return [video['video_id'], filename + "." + stream.mime_type.split("/")[1], stream.resolution, stream.fps, 0.0, dl.length]

       except Exception as ಠ_ಠ:
           print(f"Exception {ಠ_ಠ} while downloading: {url} ...retrying {i+1}/{retries}")
        
    return None

def move_files(source, dest, override = True): 
    files = os.listdir(source)

    for f in files:
        dest_file = os.path.join(dest,f)
        if os.path.exists(dest_file):
            if override: 
                os.remove(dest_file)
            else: 
                continue
        shutil.move(os.path.join(source,f), dest)        

def download_by_keyword(keyword, max_results= 5,dir_name = None, path ="./tmp", retries = 5, max_workers = 8, override_existing = False,min_res="240p", max_res="360p"): 
    san_key = keyword.strip().replace(" ", "_") if not dir_name else dir_name
    path = os.path.join(path, san_key)
    
    index = 0

    if not os.path.exists(path): 
        os.makedirs(path)
    else: 
        index = len([f for f in os.listdir(path) if str(f).endswith(".mp4")])



    search = api.search(keyword, max_results = max_results, retries=retries)
    print(f"Found {len(search)} videos matching \"{keyword}\"")
    #store downloading files in tmp dir and only copy to final dir once they are downloaded and saniizted
    dl_path = tempfile.TemporaryDirectory(dir  =  "./") 
    existing = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for result in search: 
            
            video_file = os.path.join(path, result['video_id'] + ".mp4")
            if os.path.exists(video_file): 
                existing += 1
                if override_existing:
                    os.remove(video_file)
                else:
                    continue
            
            futures.append(executor.submit(download_video, result, dl_path.name, retries=retries,min_res=min_res, max_res=max_res))

            #download(video_url, dl_path.name)

        results = [future.result() for future in concurrent.futures.as_completed(futures) if future.result()]

        df = pd.DataFrame(results, columns = ['video_id', 'file_name', 'resolution', 'fps', 'start', 'end'])

        meta_file = os.path.join(path, "info.csv")
        write_header = True
        if os.path.exists(meta_file): 
            write_header = False
            df = pd.concat([pd.read_csv(meta_file),df], keys='video_id')

        df.to_csv(meta_file, header = write_header, mode = 'a', index = False)
        print(f"Successful downloads: {len(results)}/{len(search)} Already existed: {existing}")    

    move_files(dl_path.name, path)
    dl_path.cleanup()




def download(url, path, filename = None, retries = 3): 

    for i in range(retries):
        try: 
            dl = Downloader(url)

            #todo: format
            stream = dl.streams.first()

            stream.download(path, filename = filename)
            return True

        except Exception as e:
            
            print(f"Exception {e} while downloading: {url} ...retrying {i+1}/{retries}")
        
    return False


if __name__ == "__main__":

    #download_by_keyword("debug")
    key = sys.argv[1]
    download_by_keyword(key, max_results= 1000,dir_name = None, path ="./", retries = 5, max_workers = 5, override_existing = False,min_res="240p", max_res="360p")
