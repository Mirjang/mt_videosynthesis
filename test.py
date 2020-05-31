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

from util.video_output import VideoOutput

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

    if opt.id_mapping: 
        id_mapping = list(opt.id_mapping.split(","))
        opt.id_mapping = list(map(int, id_mapping)) 
        print("using mapping: " + str(opt.id_mapping))

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
    print('#test objects = %d' % opt.nObjects)

    video = True

    print('>>> create model <<<')
    model = create_model(opt)
    print('>>> setup model <<<')
    model.setup(opt)
    #save_tensor_image(model.texture.data[0:1,0:3,:,:], 'load_test1.png')

    # create a website    
    print('>>> create a website <<<')
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if video: 
        video_output = VideoOutput(opt)

    sum_time = 0
    total_runs = dataset_size
    warm_up = 50

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
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
        
        model.test()


        b = time.perf_counter()

        if i > warm_up:  # give torch some time to warm up
            sum_time += ((b-a) * 1000)

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 10 == 0:
            print(opt.name + ":")
            print('processing (%04d)-th image... %s' % (i, img_path))
            
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        if video: 
            video_output.writeFrame(visuals)


    print('mean eval time: ', (sum_time / (total_runs - warm_up)))
    if video: 
        video_output.close()
    # save the website
    webpage.save()
    print("DONE: " + opt.name)
