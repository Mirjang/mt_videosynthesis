import os.path
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
from PIL import Image
import OpenEXR
import glob
from util.exr import channels_to_ndarray
from util.util import eulerAnglesToRotationMatrix
import cv2
import random

#for loading binary data
# def make_dataset(dir):
#     rgb = []
#     uv = [] 
#     mask = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#             if any(fname.endswith(extension) for extension in ['rgb.bin', 'rgb.BIN']):
#                 path = os.path.join(root, fname)
#                 rgb.append(path)
#             elif any(fname.endswith(extension) for extension in ['uv.bin', 'uv.BIN']):
#                 path = os.path.join(root, fname)
#                 uv.append(path)
#             elif any(fname.endswith(extension) for extension in ['mask.bin', 'mask.BIN']):
#                 path = os.path.join(root, fname)
#                 mask.append(path)
#     paths = zip(rgb,uv,mask)
#     return paths

def make_dataset(dir, opt):
    paths = [] 
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    with open(os.path.join(dir, "object_names.txt")) as objnames: 
        opt.nObjects  = len(objnames.readlines())

    i = 0  
    while os.path.exists(os.path.join(dir, str(i) + "_rgb.exr")):
        rgb = os.path.join(dir, str(i) + "_rgb.exr")
        uvs = [] 
        for l in range(opt.num_depth_layers): 
            uvs.append(os.path.join(dir, str(i) + "_uv_"+str(l)+".exr"))
        #too slow for large datasets
        #uvs = sorted(glob.glob(os.path.join(dir, str(i) + "_uv_*.exr")))
        paths.append((i, rgb, sorted(uvs)))
        i = i+1

    return paths

def loadRGBAFloatEXR(path, channel_names = ['R', 'G', 'B']): 
    assert(OpenEXR.isOpenExrFile(path), "INVALID PATH" +str(path))

    exr_file = OpenEXR.InputFile(path)
    nparr = channels_to_ndarray(exr_file, channel_names)
    exr_file.close()
    nparr = np.clip(nparr, 0.0, 1.0)
    
    #rgb = np.transpose(rgb, (1,2,0))
    #rgb = rgb[:,:, :3]
    #rgb = np.flip(rgb, 0)

    return nparr


class TransparentDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt))
        assert(opt.resize_or_crop == 'resize_and_crop')
        self.extrinsics = np.loadtxt(os.path.join(self.dir_AB, "camera_pose.txt"))
        self.IMG_DIM_X = opt.fineSize
        self.IMG_DIM_Y = opt.fineSize
        #self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.device = torch.device('cpu')
        print("DataLoader using: " + str(self.device))
        opt.update_world_pos = False
        self.update_world_pos = False
        if opt.id_mapping: 
            opt.nObjects = len(opt.id_mapping)
        self.nObjects = opt.nObjects 
        worldPositions = [0] * opt.nObjects
        dims = None
        for i in range(opt.nObjects): 
            if opt.id_mapping: 
                if opt.id_mapping[i] >= 0:
                    worldPositions[opt.id_mapping[i]] = transforms.ToTensor()(loadRGBAFloatEXR(os.path.join(self.dir_AB, "positions_"+str(i)+".exr"), channel_names=['R', 'G', 'B']))
            else: 
                worldPositions[i] = transforms.ToTensor()(loadRGBAFloatEXR(os.path.join(self.dir_AB, "positions_"+str(i)+".exr"), channel_names=['R', 'G', 'B']))
        for i in range(opt.nObjects): 
            #print(torch.is_tensor(worldPositions[i]))
            if not torch.is_tensor(worldPositions[i]):
                worldPositions[i] = torch.zeros_like(worldPositions[0])
        self.worldPositions = (torch.stack(worldPositions,0) -0.5) * 100 #undo normalisation? 
        pose_path = os.path.join(self.dir_AB, "object_pose.txt")
        if os.path.exists(pose_path): 
            print("Found custom pose file, assuming positions given in local coords")
            opt.update_world_pos = True 
            self.update_world_pos = True
            poses = np.loadtxt(pose_path)
            self.poses = np.reshape(poses, (-1, opt.nObjects - 1, 6))
            
        self.is_train = hasattr(opt, "lr")
        self.pad_front = opt.pad_front
        self.num_depth_layers = opt.num_depth_layers
        if self.pad_front: 
            opt.num_depth_layers += 1
        
    def __getitem__(self, index):
        #print('GET ITEM: ', index)
        AB_path = self.AB_paths[index]

        _, rgb_path, uv_paths = AB_path

        assert(len(uv_paths) >= self.num_depth_layers), "len(uv_paths) !>= num_depth_layers"
        # default image dimensions
        

        # load image data
        #assert(IMG_DIM == self.opt.fineSize)
        rgb_array = loadRGBAFloatEXR(rgb_path, ['R', 'G', 'B'])

        uv_arrays = []
        mask_arrays = [] 

        for i in range(self.num_depth_layers):
            mask_tmp = transforms.ToTensor()(loadRGBAFloatEXR(uv_paths[i] ,channel_names=['B'])).to(self.device)
            mask_tmp = mask_tmp * 255
            mask_tmp[mask_tmp == 255] = 0
            mask_arrays.append(mask_tmp.round().int())
            uv = transforms.ToTensor()(loadRGBAFloatEXR(uv_paths[i], channel_names=['R', 'G'])).to(self.device)
            uv_mask = torch.cat([mask_tmp, mask_tmp], 0)
            uv[uv_mask == 0] = 0 #rendering forces background to be 1, however here 0 is preferable
            uv_arrays.append(uv)

        if self.pad_front: 
            pad_uv = torch.zeros_like(uv_arrays[0])
            pad_mask = torch.zeros_like(mask_arrays[0])
            if self.is_train:# randomly insert 0s
                l = np.random.randint(0, self.num_depth_layers)
                uv_arrays.insert(l, pad_uv)
                mask_arrays.insert(l, pad_mask)
            else: 
                uv_arrays.append(pad_uv)
                mask_arrays.append(pad_mask)
        UV = torch.cat(uv_arrays, 0)
        MASK = torch.cat(mask_arrays, 0)

            

        TARGET = transforms.ToTensor()(rgb_array.astype(np.float32))
        if not self.opt.target_downsample_factor == 1: 
            TARGET = F.interpolate(TARGET.unsqueeze(0), scale_factor=1/self.opt.target_downsample_factor, mode="bilinear", align_corners=True).squeeze()

        # for i in range(self.opt.num_depth_layers):
        #     mask_tmp = loadRGBAFloatEXR(uv_paths[i],channel_names=['B'])
        #     mask_tmp = mask_tmp * 255
        #     mask_tmp[mask_tmp==255] = 0
        #     mask_arrays.append( np.rint(mask_tmp).astype(np.int32))
        #     uv = loadRGBAFloatEXR(uv_paths[i], channel_names=['R','G'])
        #     uv_mask = np.concatenate([mask_tmp,mask_tmp], axis=2)
        #     uv[uv_mask==0] = 0 #rendering forces background to be 1, however here 0 is preferable
        #     uv_arrays.append( uv )

        # uvs = np.concatenate(uv_arrays, axis=2)
        # masks = np.concatenate(mask_arrays, axis=2)

        # TARGET = transforms.ToTensor()(rgb_array.astype(np.float32))
        # UV = transforms.ToTensor()(uvs.astype(np.float32))
        # MASK = transforms.ToTensor()(masks.astype(np.int32))




        TARGET = 2.0 * TARGET - 1.0
        UV = 2.0 * UV - 1.0


        #################################
        ####### apply augmentation ######
        #################################
        # if not self.opt.no_augmentation:
        #     # random dimensions
        #     new_dim_x = np.random.randint(int(IMG_DIM_X * 0.75), IMG_DIM_X+1)
        #     new_dim_y = np.random.randint(int(IMG_DIM_Y * 0.75), IMG_DIM_Y+1)
        #     new_dim_x = int(np.floor(new_dim_x / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
        #     new_dim_y = int(np.floor(new_dim_y / 64.0) * 64 )
        #     if new_dim_x > IMG_DIM_X: new_dim_x -= 64
        #     if new_dim_y > IMG_DIM_Y: new_dim_y -= 64

        #     # random pos
        #     if IMG_DIM_X == new_dim_x: offset_x = 0
        #     else: offset_x = np.random.randint(0, IMG_DIM_X-new_dim_x)
        #     if IMG_DIM_Y == new_dim_y: offset_y = 0
        #     else: offset_y = np.random.randint(0, IMG_DIM_Y-new_dim_y)

        #     # select subwindow
        #     TARGET = TARGET[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
        #     UV = UV[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]



        # else:
        #     new_dim_x = int(np.floor(IMG_DIM_X / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
        #     new_dim_y = int(np.floor(IMG_DIM_Y / 64.0) * 64 )
        #     offset_x = 0
        #     offset_y = 0
        #     # select subwindow
        #     TARGET = TARGET[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
        #     UV = UV[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]


        #################################
        if self.update_world_pos: 
            world_positions = self.worldPositions.clone().detach().numpy()
            _, _, W, H = world_positions.shape
            for i in range(1, self.nObjects-1): 
                pose = self.poses[index, i-1]
                r = eulerAnglesToRotationMatrix(np.deg2rad(pose[3:6]))
                t = pose[:3, np.newaxis]
                world_positions[i] = (np.dot(r, world_positions[i].reshape(3, -1)) + t).reshape(3,W,H)
            wp = torch.tensor(world_positions)
        else: 
            wp = self.worldPositions
        extrinsics = torch.tensor(self.extrinsics[index].astype(np.float32))[:3,...]
        return {'TARGET': TARGET, 'UV': UV, 'MASK': MASK,
                'paths': rgb_path, 'extrinsics' : extrinsics, 'worldpos': wp}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'TransparentDataset'