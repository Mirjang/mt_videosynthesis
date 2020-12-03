import time
import os
from options.train_options import TrainOptions
from data import CreateDataLoader, SplitDataLoader
from models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
from PIL import Image
from sanity_check import sanity_check
import copy

if __name__ == '__main__':
    opt = TrainOptions().parse()

    abort_file = "/mnt/raid/patrickradner/kill" + str(opt.gpu_ids[0]) if len(opt.gpu_ids)>0 else "cpu"
    if os.path.exists(abort_file): 
        os.remove(abort_file)
        exit("Abort using file: " + abort_file)

    if opt.sanity_check: 
        sanity_check(opt)

    batch_size = opt.batch_size
    parallell_batch_size = opt.parallell_batch_size if opt.parallell_batch_size > 0 else opt.batch_size
    opt.batch_size = parallell_batch_size
    n_acc_batches = batch_size // parallell_batch_size
    opt.n_acc_batches = n_acc_batches
    assert batch_size % parallell_batch_size == 0, "Batch size should be divisible by parallell batch size"
    
    data_loader = CreateDataLoader(copy.deepcopy(opt))

    validation_size=0
    if opt.validation_freq>0:
        opt_val = copy.deepcopy(opt)
        opt_val.phase = opt.validation_set
        opt_val.max_dataset_site = opt.max_val_dataset_size
        if opt.validation_set == "split": 
            torch.manual_seed(42)
            validation_loader, data_loader = SplitDataLoader(data_loader, copy.deepcopy(opt), opt_val, length_first = opt.max_val_dataset_size)
        else:
            validation_loader = CreateDataLoader(opt_val)
        validation_set = validation_loader.load_data()
        validation_size = len(validation_loader)

    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training samples = %d' % dataset_size)
    print('#validation samples = %d' % validation_size)

    model = create_model(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        torch.cuda.empty_cache()

        verbose = True
        #validation
        if (opt.validation_freq>0 and epoch % opt.validation_freq == 0):# or epoch == 1: # do val first so we dont get nasty crashes after hours of training
            iter_start_time = time.time()
            val_losses = {}
            visuals = {}
            for i, data in enumerate(validation_set):
                model.set_input(data) 
                torch.cuda.synchronize()                
                model.test()
                current_losses, visuals = model.compute_validation_losses()
                #current_losses = model.get_current_losses()
                # avg. validation loss
                val_losses = {key: val_losses.get(key, 0) + current_losses.get(key, 0) / validation_size
                    for key in set(val_losses) | set(current_losses)}

            message = 'VALIDATION (epoch: %d): ' % (epoch)
            for k, v in val_losses.items():
                message += '%s: %.6f ' % (k, v)
            print(message)
            visualizer.reset()
            t = (time.time() - iter_start_time) / opt.batch_size
            #visualizer.print_current_losses(epoch, epoch_iter, val_losses, t, t_data)
            if opt.display_id > 0:
                visualizer.plot_validation_losses(epoch, opt, val_losses)
                save_result = False
                visualizer.display_current_results(visuals, epoch, save_result)

            torch.cuda.empty_cache()
        # training loop

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        losses={}
        i = 0
        for data in dataset:
            i+=1
            if os.path.exists(abort_file): 
                exit("Abort using file: " + abort_file)
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
  
            ################## training code ##################
            model.set_input(data)
            model.compute_losses(epoch, verbose = verbose)

            if (i+1) % n_acc_batches == 0 or n_acc_batches == 0 : # accumulate gradients for less noisy updates
                model.optimize_parameters(epoch, verbose = verbose)

            ################## end training code ##################

            verbose = False or opt.verbose
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            if total_steps % (opt.print_freq) == 0:
                print(opt.name)
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            
            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
