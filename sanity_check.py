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


    #opt.max_dataset_size = 1
    opt.batchsize = 32
    opt.print_freq = 250
    opt.display_freq = 100
    opt.update_html_freq = 100
    opt.validation_freq = 1
    abort_file = "/mnt/raid/patrickradner/kill" + str(opt.gpu_ids[0]) if len(opt.gpu_ids)>0 else "cpu"

    if os.path.exists(abort_file): 
        os.remove(abort_file)
        exit("Abort using file: " + abort_file)

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

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):


        #validation
        if epoch % opt.validation_freq == 0 and opt.validation_freq>0:
            iter_start_time = time.time()
            val_losses = {}
            for i, data in enumerate(validation_set):
                model.set_input(data) 
                torch.cuda.synchronize()                
                model.test()
                model.compute_losses()
                current_losses = model.get_current_losses()
                # avg. validation loss
                val_losses = {key: val_losses.get(key, 0) + current_losses.get(key, 0) / validation_size
                    for key in set(val_losses) | set(current_losses)}

            visualizer.reset()
            val_losses = {"val_" + k : v for k, v in val_losses.items() }
            t = (time.time() - iter_start_time) / opt.batch_size
            print("validation:" + str(val_losses))
            #visualizer.print_current_losses(epoch, epoch_iter, val_losses, t, t_data)
            if opt.display_id > 0:
                visualizer.plot_validation_losses(epoch, opt, val_losses)
                save_result = False
                print(model.get_current_visuals().items())
                visuals = {"val_" + k : v for k,v in model.get_current_visuals()}
                visualizer.display_current_results(visuals, epoch, save_result)


        # training loop

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        losses={}
        for i, data in enumerate(dataset):
            if os.path.exists(abort_file): 
                exit("Abort using file: " + abort_file)


            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_input(data)
            model.optimize_parameters(epoch)

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

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)



        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
