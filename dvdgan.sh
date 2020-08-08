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
# DATASET=cifar10
# DATASET_MODE=dummy
DATASET=UCF101
DATASET_MODE=ucf101


# models
MODEL=dvdgan
NAME=nobndvdgan
DISPNAME=${NAME}_screens

# optimizer parameters
LR=0.0004
BATCHSIZE=8
RESOLUTION=64
FPS=25

# vid settings
SKIP=1
LEN=1.0

FREQ=500
#--verbose --sanity_check
python train.py --niter 250 --continue_train --epoch 220 --epoch_count 0 --niter_decay 250 --train_mode "mixed" --clip_grads .5 --n_critic 2 --lambda_L1 20 --lambda_S 1 --lambda_T 2 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 1 --validation_set val --pretrain_epochs 0 --tlg .2 --tld .8 --init_type xavier --init_gain .0002 --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --print_freq $FREQ --display_freq $FREQ
#python test.py --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
