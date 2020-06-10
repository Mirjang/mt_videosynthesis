set -ex

VISDOM_PORT=8197

DATASETS_DIR=/mnt/raid/patrickradner/checkpoints
CHECKPOINT_DIR=/mnt/raid/patrickradner/datasets

DATASET=movingmnist

# models
MODEL=simpleVideo

# optimizer parameters
LR=0.001

# GPU
GPU_ID=3

python sanity_check.py --niter 2000 --dataroot $DATASETS_DIR/$DATASET --name sanity_check --model $MODEL --dataset_mode movingmnist --norm batch --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT 

#python test.py --fix_renderer --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0