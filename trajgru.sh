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

DATASETS_DIR=/mnt/raid/patrickradner/datasets/yt
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints
RESULTS_DIR=/mnt/raid/patrickradner/results

#DATASET=movingmnist
#DATASET=3hr
# DATASET=cifar10
# DATASET_MODE=dummy
DATASET=river_relaxing
#DATASET=UCF101

DATASET_MODE=video
VAL_SET=split


# models
MODEL=dvdgan
# optimizer parameters
LR=0.0003
BATCHSIZE=256
SUB_BATCH=2
RESOLUTION=128
FPS=25
GENERATOR=trajgru

NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_aux_seg
git add -A 
git commit -m $NAME
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1.0

FREQ=200
DISP_FRAMES=16
VAL_SIZE=100
#--verbose --sanity_check
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --use_segmentation --niter 1 --niter_decay 250 --lr_policy cosine --train_mode "mixed" --clip_grads .05 --n_critic 3 --lambda_L1 20 --lambda_S .1 --lambda_T 1 --lambda_GP 1 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set $VAL_SET --pretrain_epochs 0 --tlg .3 --tld .7 --max_val_dataset_size $VAL_SIZE --init_type orthogonal --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids 0 --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#python test.py --grid 5 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 

# DATASET=gaugan
# DATASET_MODE=image
# python test.py --grid 5 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 
