import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
from PIL import Image

IMG_DIM = 512

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.bin', '.BIN']):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def load_intrinsics(input_dir, id):
    file = open(input_dir+"/"+str(id)+"_intrinsics.txt", "r")
    intrinsics = [[(float(x) for x in line.split())] for line in file]
    file.close()
    intrinsics = list(intrinsics[0][0])
    return intrinsics

def load_rigids(input_dir, id):
    file = open(input_dir+"/"+str(id)+"_rigid.txt", "r")
    rigid_floats = [[float(x) for x in line.split()] for line in file] # note that it stores 5 lines per matrix (blank line)
    file.close()
    all_rigids = [ [rigid_floats[4*idx + 0],rigid_floats[4*idx + 1],rigid_floats[4*idx + 2],rigid_floats[4*idx + 3]] for idx in range(0, len(rigid_floats)//4) ]
    return all_rigids

def make_ids(paths, root_dir):
    object_ids = []
    ids = []
    
    for fname in paths:
        s = fname.rfind('/')
        n = fname[s+1:-4]
        l = n.rfind('_')
        object_id = int(n[0:l])
        i = int(n[l+1:])
        ids.append(i)
        object_ids.append(object_id)

    return (object_ids, ids)

class MultiDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.AB_object_ids, self.AB_ids = make_ids(self.AB_paths, self.root)
        
        objects = set(self.AB_object_ids)
        for i in range(0, len(self.AB_object_ids)):
            cnt = 0
            for id in objects:
                if self.AB_object_ids[i] == id:
                    self.AB_object_ids[i] = cnt
                cnt += 1

        self.intrinsics = []
        self.extrinsics = []

        for id in objects:
            self.intrinsics.append(load_intrinsics(self.dir_AB, id))
            self.extrinsics.append(load_rigids(self.dir_AB, id))

        print('objects: ', objects)

        opt.nObjects = len(objects)

        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        #print('GET ITEM: ', index)
        AB_path = self.AB_paths[index]
        image_id = self.AB_ids[index]
        object_id = self.AB_object_ids[index]

        intrinsics = self.intrinsics[object_id]
        extrinsics = self.extrinsics[object_id][image_id]

        # default image dimensions
        IMG_DIM_X = 512
        IMG_DIM_Y = 512

        # load image data
        #assert(IMG_DIM == self.opt.fineSize)
        img_array = np.memmap(AB_path, dtype='float32', mode='r').__array__()
        if img_array.size != IMG_DIM_X * IMG_DIM_Y * 5:
            IMG_DIM_X = int(img_array[0])
            IMG_DIM_Y = int(img_array[1])
            img_array = img_array[2:]
            intrinsics = img_array[0:4]
            img_array = img_array[4:]

        img_array = np.clip(img_array, 0.0, 1.0)
        img = np.resize(img_array,  (IMG_DIM_Y, IMG_DIM_X, 5))
        A = img[:,:,0:3]
        B = img[:,:,3:5]
        B = np.concatenate((B, np.zeros((IMG_DIM_Y, IMG_DIM_X, 1))), axis=2)

        TARGET = transforms.ToTensor()(A.astype(np.float32))
        UV = transforms.ToTensor()(B.astype(np.float32))

        TARGET = 2.0 * TARGET - 1.0
        UV = 2.0 * UV - 1.0


        #################################
        ####### apply augmentation ######
        #################################
        if not self.opt.no_augmentation:
            # random dimensions
            new_dim_x = np.random.randint(int(IMG_DIM_X * 0.75), IMG_DIM_X+1)
            new_dim_y = np.random.randint(int(IMG_DIM_Y * 0.75), IMG_DIM_Y+1)
            new_dim_x = int(np.floor(new_dim_x / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
            new_dim_y = int(np.floor(new_dim_y / 64.0) * 64 )
            if new_dim_x > IMG_DIM_X: new_dim_x -= 64
            if new_dim_y > IMG_DIM_Y: new_dim_y -= 64

            # random pos
            if IMG_DIM_X == new_dim_x: offset_x = 0
            else: offset_x = np.random.randint(0, IMG_DIM_X-new_dim_x)
            if IMG_DIM_Y == new_dim_y: offset_y = 0
            else: offset_y = np.random.randint(0, IMG_DIM_Y-new_dim_y)

            # select subwindow
            TARGET = TARGET[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
            UV = UV[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]

            # compute new intrinsics
            # TODO: atm not needed but maybe later

            ## rotate 0,90,180,270
            #rot = np.random.randint(0,4)
            #if rot == 1:
            #    TARGET.transpose_(1, 2)
            #    UV.transpose_(1, 2)
            #elif rot == 2:
            #    TARGET.flip_(2)
            #    UV.flip_(2)
            #elif rot == 3:
            #    TARGET.flip_(2)
            #    UV.flip_(2)
            #    TARGET.transpose_(1, 2)
            #    UV.transpose_(1, 2)

        #################################

        return {'TARGET': TARGET, 'UV': UV,
                'paths': AB_path,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'object_id': object_id}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'MultiDataset'
