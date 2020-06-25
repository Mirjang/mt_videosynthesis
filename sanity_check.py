import time
import os
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    opt = TrainOptions().parse()
    sanity_check(opt)

def sanity_check(opt):
    abort_file = "/mnt/raid/patrickradner/kill" + str(opt.gpu_ids[0]) if len(opt.gpu_ids)>0 else "cpu"

    if os.path.exists(abort_file): 
        os.remove(abort_file)
        exit("Abort using file: " + abort_file)


    #opt.max_dataset_size = 1
    freq = 20
    opt.batch_size = 1
    opt.print_freq = freq
    opt.display_freq = freq
    opt.update_html_freq = freq
    opt.validation_freq = 50
    opt.niter = 500
    opt.niter_decay = 0 
    opt.display_env = "sanity_check"
    opt.num_display_frames = int(opt.max_clip_length*opt.fps)-2
    opt.train_mode ="frame"
    #opt.lr = 0.004
    opt.pretrain_epochs = 0

    opt.verbose = True

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    #show some data using opencv, only works when display is available
    # for _,data in enumerate(dataset): 
    #     clip = data['VIDEO'][0] #first elem in batch
    #     T,_,_,_ = clip.shape
    #     print(T)
    #     for i in range(min(T,150)):
    #         frame = clip[i].numpy()#.transpose(1,2,0)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         cv2.imshow("1", frame)
    #         cv2.waitKey(int(1.0/float(30)*1000))
    #     break

    if opt.validation_freq>0:
        phase = opt.phase
        opt.phase = opt.validation_set
        validation_loader = CreateDataLoader(opt)
        validation_set = validation_loader.load_data()
        opt.phase = phase

        validation_size = len(validation_loader)
        print('#training samples = %d' % dataset_size)
        print('#validation samples = %d' % validation_size)

    model = create_model(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_steps = 0

    data = next(iter(dataset))


    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        # training loop

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        losses={}
            

        if os.path.exists(abort_file): 
            exit("Abort using file: " + abort_file)


        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        visualizer.reset()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        
        model.set_input(data)
        model.optimize_parameters(epoch, verbose = opt.verbose)

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            print(opt.name)
            losses = model.get_current_losses()
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

        iter_data_time = time.time()


        if epoch % 50 == 0:
            print('End of sanity_check epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
