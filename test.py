import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
from PIL import Image
import time
import re

import torchvision

def save_tensor_image(input_image, image_path):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

if __name__ == '__main__':
    opt = TestOptions().parse()

    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_augmentation = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.num_threads = 0
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#test images = %d' % dataset_size)


    print('>>> create model <<<')
    model = create_model(opt)
    print('>>> setup model <<<')
    model.setup(opt)
    #save_tensor_image(model.texture.data[0:1,0:3,:,:], 'load_test1.png')

    # create a website    
    print('>>> create a website <<<')
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    sum_time = 0
    total_runs = dataset_size
    warm_up = 50
    block = 0
    # test with eval mode. This only affects layers like batchnorm and dropout.
    out_buffer = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
    #for i in range(len(dataset)):

        #data = dataset[i]
        #if i >= opt.num_test:
        #    break
        model.set_input(data)

        torch.cuda.synchronize()
        a = time.perf_counter()
        
        model.test() #forward pass no grad


        b = time.perf_counter()

        if i > warm_up:  # give torch some time to warm up
            sum_time += ((b-a) * 1000)

        vids = model.predicted_video.detach().cpu()
        out_buffer.append(*torch.split(vids, 1, dim=0))

       #visuals = model.get_current_visuals()
       # img_path = model.get_image_paths()
        if i % 10 == 0:
            print(opt.name + ":")
            print('processing (%04d)-th sample...' % (i))
        
      #  save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        if len(out_buffer) >= opt.grid**2: 
            

            grid = out_buffer[:opt.grid**2]
            out_buffer = out_buffer[opt.grid**2: ]
            rows = []
            for x in range(opt.grid): 
                a = x*opt.grid
                b = a + opt.grid
                row = torch.cat(grid[a:b], dim = -2)
                rows.append(row)

            out = torch.cat(rows, dim = -1).squeeze(0).permute(0,2,3,1) * 255 # T,W,H,C
            torchvision.io.write_video(os.path.join(web_dir, f"fake_block{block}.mp4"), out, opt.fps / 2)
            block += 1

    print('mean eval time: ', (sum_time / (total_runs - warm_up)))
    # save the website
    print("DONE: " + opt.name)
