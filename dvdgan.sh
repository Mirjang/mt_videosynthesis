set -ex

# GPU
GPU_ID=3

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

NAME=DVD_GAN

#DATASET=movingmnist
#DATASET=3hr
#DATASET=cifar10
DATASET=UCF101
DATASET_MODE=ucf101


# models
MODEL=dvdgan

NAME=ucf101

# optimizer parameters
LR=0.0004
BATCHSIZE=8
RESOLUTION=64
FPS=25

FREQ=500
#--verbose --sanity_check
python train.py --niter 500 --niter_decay 250 --train_mode "mixed" --clip_grads .5 --lambda_S .1 --lambda_T 2. --max_clip_length 1 --skip_frames 1 --validation_freq 2 --validation_set split --pretrain_epochs 0 --tlg .1 --tld .8 --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $NAME --update_html_freq $FREQ --print_freq $FREQ --display_freq $FREQ
#python test.py --fix_renderer --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
