set -ex

# GPU
GPU_ID=2

if [[ $(nvidia-smi | grep "^|    $GPU_ID    ") ]]; then
    read -p "GPU currently in use, continue? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]
    then
        exit 1
    fi
fi

VISDOM_PORT=8197

DATASETS_DIR=/mnt/raid/patrickradner/datasets
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints

#DATASET=movingmnist
#DATASET=3hr
DATASET=cifar10

# models
MODEL=simpleVideo

# optimizer parameters
LR=0.0004


python sanity_check.py --niter 100 --niter_decay 50 --train_mode "mixed" --max_clip_length 1 --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name sanity_check --model $MODEL --dataset_mode dummy --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT
#python test.py --fix_renderer --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
