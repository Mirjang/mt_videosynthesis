set -ex

DATASETS_DIR=../datasets
DATASET=3hr

# models
MODEL=simpleVideo

# optimizer parameters
LR=0.001

# GPU
GPU_ID=-1

python sanity_check.py --niter 2000 --dataroot $DATASETS_DIR/$DATASET --name $MODEL-$DATASET --model $MODEL --dataset_mode video --norm batch --gpu_ids $GPU_ID --lr $LR

#python test.py --fix_renderer --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0