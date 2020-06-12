set -ex

VISDOM_PORT=8197

DATASETS_DIR=/mnt/raid/patrickradner/datasets
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints

#DATASET=movingmnist
DATASET=3hr

# models
MODEL=simpleVideo

# optimizer parameters
LR=0.0004

# GPU
GPU_ID=3

python sanity_check.py --niter 20000 --niter_decay 10000  --max_clip_length 2 --dataroot $DATASETS_DIR/$DATASET --name sanity_check --model $MODEL --dataset_mode video --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT
#python test.py --fix_renderer --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
