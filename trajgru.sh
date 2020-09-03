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

DATASETS_DIR=/mnt/raid/patrickradner/datasets/yt/
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints
RESULTS_DIR=/mnt/raid/patrickradner/results

#DATASET=movingmnist
#DATASET=3hr
# DATASET=cifar10
# DATASET_MODE=dummy
DATASET=river_relaxing
DATASET_MODE=video


# models
MODEL=dvdgan
# optimizer parameters
LR=0.0003
BATCHSIZE=32
SUB_BATCH=1
RESOLUTION=128
FPS=25
GENERATOR=trajgru

NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_3
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1.0

FREQ=200
DISP_FRAMES=16
VAL_SIZE=100
#--verbose --sanity_check
python train.py --continue_train --niter 250 --niter_decay 250 --train_mode "mixed" --clip_grads .05 --n_critic 3 --lambda_L1 20 --lambda_S 1 --lambda_T 5 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set val --pretrain_epochs 0 --tlg .3 --tld .7 --max_val_dataset_size $VAL_SIZE --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#python test.py --grid 5 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 
