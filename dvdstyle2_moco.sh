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
DATASETS_DIR=/mnt/raid/patrickradner/datasets/mocogan/shapes
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints
RESULTS_DIR=/mnt/raid/patrickradner/results


DATASET=1
DATASET_MODE=mocogan

# models
MODEL=dvdgan
# optimizer parameters
LR=0.0003
BATCHSIZE=8
SUB_BATCH=4
RESOLUTION=64
FPS=16
GENERATOR=style2

NAME=moco_${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_base


DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=2
CLIP=2.5

FREQ=100
DISP_FRAMES=16
VAL_SIZE=100
VAL_SET=split
#--verbose --sanity_check
python train.py --ch_g 32 --ch_ds 32 --ch_dt 32 --no_dt_prepool --start_fp 2 --max_fp 5 --lr_policy cosine --style_noise --niter 10 --niter_decay 250 --clip_grads $CLIP --n_critic 3 --lambda_S 1 --lambda_T 5 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set $VAL_SET --max_val_dataset_size $VAL_SIZE --init_type None --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#CUDA_VISIBLE_DEVICES=3 python test.py --epoch 5 --grid 1 --ch_g 32 --ch_ds 16 --ch_dt 64 --no_dt_prepool --conditional --start_fp 2 --max_fp 5 --up_blocks_per_rnn 1 --gru_layers 1 --phase $VAL_SET --max_dataset_size $VAL_SIZE --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 

DATASET=gaugan
DATASET_MODE=image
#python test.py --grid 5 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 
