set -ex

# GPU
GPU_ID=-1

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
MODEL=lhc
# optimizer parameters
LR=0.0003
BATCHSIZE=1
SUB_BATCH=1
RESOLUTION=128
FPS=25
GENERATOR=lhc

NAME=debug
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1.0

FREQ=1
DISP_FRAMES=16
VAL_SIZE=100
#--verbose --sanity_check
#CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --niter 250 --niter_decay 250 --use_segmentation --train_mode "mixed" --clip_grads .05 --n_critic 3 --lambda_L1 20 --lambda_S 1 --lambda_T 1 --lambda_GP 10 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set val --pretrain_epochs 0 --tlg .3 --tld .7 --max_val_dataset_size $VAL_SIZE --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids 0 --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#cant run cpu w/ upper command
#python train.py --niter 250 --verbose --niter_decay 250 --use_segmentation --train_mode "mixed" --clip_grads .05 --n_critic 3 --lambda_L1 20 --lambda_S 1 --lambda_T 1 --lambda_GP 10 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set val --pretrain_epochs 0 --tlg .3 --tld .7 --max_val_dataset_size $VAL_SIZE --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ

python debug.py --niter 250 --verbose --niter_decay 250 --use_segmentation --no_augmentation --train_mode "mixed" --clip_grads .05 --n_critic 3 --lambda_L1 20 --lambda_S 1 --lambda_T 1 --lambda_GP 10 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set val --pretrain_epochs 0 --tlg .3 --tld .7 --max_val_dataset_size $VAL_SIZE --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ


#python test.py --grid 5 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 
