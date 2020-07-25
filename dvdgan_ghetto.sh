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
NAME=ghettogan

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
python train.py --niter 500 --niter_decay 250 --unroll_frames 5 --train_mode "mixed" --clip_grads .1 --n_critic 2 --lambda_L1 100 --lambda_S 1 --lambda_T 1 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set split --pretrain_epochs 1 --tlg .2 --tld .8 --init_type orthogonal --init_gain .0002 --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $NAME --update_html_freq $FREQ --print_freq $FREQ --display_freq $FREQ
#python test.py --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
