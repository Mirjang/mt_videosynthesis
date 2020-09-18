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
DATASETS_DIR=/mnt/raid/patrickradner/datasets/yt/
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints
RESULTS_DIR=/mnt/raid/patrickradner/results

#DATASET=movingmnist
# DATASET=cifar10
# DATASET_MODE=dummy
DATASET=river_relaxing
DATASET_MODE=video

# models
MODEL=dvdgan
# optimizer parameters
LR=0.0001
BATCHSIZE=128
SUB_BATCH=2
RESOLUTION=128
FPS=25
GENERATOR=dvdgan

NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_noise_cgan
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1.0

FREQ=500
DISP_FRAMES=16
VAL_SIZE=100
VAL_SET=val #split
#--verbose --sanity_check
python train.py --niter 1 --niter_decay 250 --train_mode "mixed" --lr_policy cosine --clip_grads .05 --n_critic 3 --lambda_L1 50 --lambda_S 1 --lambda_T 1 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set $VAL_SET --pretrain_epochs 0 --tlg .3 --tld .7 --max_val_dataset_size $VAL_SIZE --init_type orthogonal --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#python test.py --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
