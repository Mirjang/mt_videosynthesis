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
#DATASET=3hr
#DATASET=cifar10
# DATASET_MODE=dummy
# DATASET=ucf101
DATASET=river_relaxing
DATASET_MODE=video


# models
MODEL=lhc
# optimizer parameters
LR=0.003
BATCHSIZE=32
SUB_BATCH=2
RESOLUTION=64
FPS=25
GENERATOR=lhc

NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_debug
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1
NCRITIC=3

FREQ=100
DISP_FRAMES=16
#--verbose --sanity_check
# #debug
# LR=0.3
# BATCHSIZE=8
# SUB_BATCH=8
# FPS=10
#FREQ=10
# NCRITIC=1
#CUDA_VISIBLE_DEVICES=${GPU_ID}  python train.py --verbose --max_dataset_size 1 --niter 250 --niter_decay 250 --train_mode "mixed" --clip_grads 10 --n_critic 1 --lambda_L1 1 --lambda_S 0 --lambda_T 0 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set val --pretrain_epochs 100 --tlg .2 --tld .8 --max_val_dataset_size 500 --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH  --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids 0 --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --niter 250 --niter_decay 250 --train_mode "mixed" --clip_grads .05 --n_critic $NCRITIC --lambda_L1 100 --lambda_S 0 --lambda_T 0 --pretrain_epochs 100 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 2 --validation_set val --tlg .2 --tld .8 --max_val_dataset_size 100 --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH  --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids 0 --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --niter 1 --niter_decay 250 --use_segmentation --num_segmentation_classes 1 --lr_policy cosine --train_mode "mixed" --clip_grads .5 --n_critic $NCRITIC --lambda_reg 1 --lambda_S 1 --lambda_T 1 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 2 --validation_set val --tlg .2 --tld .8 --max_val_dataset_size 100 --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH  --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids 0 --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py --grid 5 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 
