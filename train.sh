set -ex

DATASETS_DIR=../datasets
DATASET=3hr

# models
MODEL=simpleVideo

# optimizer parameters
LR=0.001

# GPU
GPU_ID=0

python train.py --niter 2000 --dataroot $DATASETS_DIR/$OBJECT --name $MODEL-$Dataset --model $MODE --dataset_mode video --norm batch --gpu_ids $GPU_ID --lr $LR

#python test.py --fix_renderer --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0