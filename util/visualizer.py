import numpy as np
import os
import sys
import ntpath
import time
from . import util
from . import html
from skimage.transform import resize
from PIL import Image
import torch.nn.functional as F


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def imresize(im, shape, interp = 'bicubic'): 
    #return np.array(Image.fromarray(np.uint8(im*256)).resize(shape))
    return resize(im.astype(float), shape)

def tensor2vid(video): 
    video = video*256
    video = video.permute(0,1,3,4,2)
    return video


# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    vids, vidtxts, vidlinks = [], [], []


    for label, im_data in visuals.items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        if label.endswith("_video"):
            vid = tensor2vid(im_data)

            vidlinks.append(image_name)
            vidtxts.append(label)
            vids.append(vid)
        else: 

            im = util.tensor2im(im_data)

            h, w, _ = im.shape
            
            height = int(width * h / float(w))
            im = imresize(im, (height,width), interp='bicubic')

            #im = imresize(im, (height,widht), interp='bicubic')
            #if aspect_ratio > 1.0:
            #    im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            #if aspect_ratio < 1.0:
            #    im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')

            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    if len(ims)>0:
        webpage.add_images(ims, txts, links, width=width)
    if len(vids)>0:
        pass

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt

        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        self.reserved_ids = 5
        self.display_ids = {}

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, aspect_ratio=1.0, width=256, prefix = ""):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                #h, w = next(iter(visuals.values())).shape[2:4] #if first visual is a video these values will be garbage
                h,w = width, width
                for label, image in visuals.items(): 
                    if not (label.endswith("_video") or label.endswith("_plt")):
                        # print(label, image.shape)
                        h, w = image.shape[2:4]
                height = int(width * h / float(w))
                h = height
                w = width
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                videos = []
                video_labels = []
                plts = []
                idx = 0
                for label, image in visuals.items():
                    #
                    if label.endswith("_video"):
                        if len(image.shape) is 5: 
                            N,*_ = image.shape
                            for v in range(min(N,4)):

                                videos.append(image[v,:,[2,1,0],...].permute(0,2,3,1))
                                video_labels.append(label + str(v))
                        else:
                            videos.append(image[:,[2,1,0],...].permute(0,2,3,1))
                            video_labels.append(label)
                    elif label.endswith("_plt"):
                        plts.append((label,image))
                    else:
                        image_numpy = util.tensor2im(image)
                        image_numpy = imresize(image_numpy, (h, w), interp='bicubic').astype(np.uint8).transpose(2,0,1)

                        images.append(image_numpy)

                    label_html_row += '<td>%s</td>' % label
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                # white_image = np.ones((3,h,w), dtype = np.uint8) * 255
                # while idx % ncols != 0:
                #     images.append(white_image)
                #     label_html_row += '<td></td>'
                #     idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
    
                try:
                    if len(images)>0:
                        
                        self.vis.images(images, nrow=ncols, win=self.display_id + 1, padding=1, opts=dict(title=title + ' images'))
                    if len(videos)>0: 
                        for (label,video) in iter(zip(video_labels,videos)): 
                            
                            if label in self.display_ids:
                                id = self.display_ids[label]
                            else: 
                                id = len(self.display_ids)
                                self.display_ids[label] = id

                            T,C,H,W = video.shape
                            if W < width:
                                video = video.permute(0,3,1,2)
                                video = F.interpolate(video, size=(h,w))
                                video = video.permute(0,2,3,1)
                            self.vis.video(video,win=self.display_id+self.reserved_ids+id,opts=dict(title=prefix + label, fps=self.opt.fps/self.opt.skip_frames // 2))
                    if len(plts)>0: 

                        for (label,plt) in iter(plts):                         
                            if label in self.display_ids:
                                id = self.display_ids[label]
                            else: 
                                id = len(self.display_ids)
                                self.display_ids[label] = id
                            opts = plt["opts"].copy()
                            opts["title"] = prefix + opts["title"]
                            self.vis.line(
                                X=plt["X"],
                                Y=plt["Y"],
                                opts=opts,
                                win=self.display_id + self.reserved_ids + id)
                                                        
                    # label_html = '<table>%s</table>' % label_html
                    # self.vis.text(table_css + label_html, win=self.display_id + 2,
                    #               opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    if label.endswith("_video"):
                        if len(image.shape) is 5: 
                            N,*_ = image.shape
                            for v in range(N):
                                self.vis.video(image[v,:,[2,1,0],...].permute(0,2,3,1),win=self.display_id + idx,opts=dict(title=label, fps=self.opt.fps/self.opt.skip_frames))
                        else:
                            self.vis.video(image[:,[2,1,0],...].permute(0,2,3,1),win=self.display_id + idx,opts=dict(title=label, fps=self.opt.fps/self.opt.skip_frames))
                        
                    else:
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy, opts=dict(title=label),
                                    win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                if label.endswith("_video") or label.endswith("_plt"):
                    pass
                else:
                    image_numpy = util.tensor2im(image)
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    if label.endswith("_video") or label.endswith("_plt"):
                        pass
                    else:
                        image_numpy = util.tensor2im(image)
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)

        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

        if len(self.plot_data['legend']) > 1:
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
            Y=np.array(self.plot_data['Y'])
        else: 
            X=np.array([self.plot_data['X']]).flatten()
            Y=np.array([self.plot_data['Y']]).flatten()

        try:
            self.vis.line(
                X=X,
                Y=Y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()


    # losses: dictionary of error labels and values
    def plot_validation_losses(self, epoch, opt, losses):
        if not hasattr(self, 'plot_val_data'):
            self.plot_val_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_val_data['X'].append(epoch)

        self.plot_val_data['Y'].append([losses[k] for k in self.plot_val_data['legend']])

        if len(self.plot_val_data['legend']) > 1:
            X=np.stack([np.array(self.plot_val_data['X'])] * len(self.plot_val_data['legend']), 1)
            Y=np.array(self.plot_val_data['Y'])
        else: 
            X=np.array([self.plot_val_data['X']]).flatten()
            Y=np.array([self.plot_val_data['Y']]).flatten()

        try:
            self.vis.line(
                X=X,
                Y=Y,
                opts={
                    'title': self.name + 'Validation loss over time',
                    'legend': self.plot_val_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id+4)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()


    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f)\t' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
