set -ex

# GPU
GPU_ID=1

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
MODEL=stylegan
# optimizer parameters
LR=0.0003
BATCHSIZE=8
SUB_BATCH=8
RESOLUTION=256
FPS=25
GENERATOR=0

NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1.0

FREQ=1
DISP_FRAMES=16
VAL_SIZE=0
VAL_SET=val #split
#--verbose --sanity_check
python train.py --niter 10 --niter_decay 250 --train_mode "mixed" --lr_policy cosine --clip_grads 10000000 --n_critic 3 --lambda_S 1 --lambda_T 1 --lambda_AUX 0 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 0 --pretrain_epochs 0 --max_val_dataset_size $VAL_SIZE --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#python test.py --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
