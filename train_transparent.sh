set -ex

DATASETS_DIR=C:/Users/Patrick/Desktop/NeuralTexture/TransparentNeuralRendering/Data/ManualCam_2019-10-28T18-28-56

# objects
#OBJECT=Globe
OBJECT=SPHERE

# renderer
RENDERER=MultiTarget-neuralRenderer_200

# models
MODEL=neuralRenderer
#MODEL=pix2pix

# optimizer parameters
LR=0.001

# GPU
GPU_ID=1

python train.py --fix_renderer --niter 2000 --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --gpu_ids $GPU_ID --lr $LR

python test.py --fix_renderer --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0