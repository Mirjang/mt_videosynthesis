set -ex

# GPU
GPU_ID=0

if [[ $(nvidia-smi | grep "^|    $GPU_ID    ") ]]; then
    read -p "GPU currently in use, continue? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]
    then
        exit 1
    fi
fi
VISDOM_PORT=8197
DATASETS_DIR=/mnt/raid/patrickradner/datasets/
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints
RESULTS_DIR=/mnt/raid/patrickradner/results

DATASET=processed_data
DATASET_MODE=videofolder

# models
MODEL=dvdgan
# optimizer parameters
LR=0.0003
BATCHSIZE=8
SUB_BATCH=8
RESOLUTION=64
FPS=25
GENERATOR=style2

NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_base_noise
# git add -A 
# git status | grep modified
# if [ $? -eq 0 ]
# then
#     git commit -m $NAME
# fi
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1
CLIP=2.5

FREQ=100
DISP_FRAMES=16
VAL_SIZE=10
VAL_SET=test #split
#--verbose --sanity_check
python train.py --ch_g 16 --ch_ds 32 --ch_dt 32 --no_dt_prepool --conditional --start_fp 2 --max_fp 4 --up_blocks_per_rnn 1 --gru_layers 1 --lr_policy cosine --niter 10 --niter_decay 250 --clip_grads $CLIP --n_critic 3 --lambda_S 1 --lambda_T 5 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set $VAL_SET --max_val_dataset_size $VAL_SIZE --init_type None --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#python test.py --ch_g 16 --ch_ds 32 --ch_dt 32 --no_dt_prepool --conditional --start_fp 2 --max_fp 5 --up_blocks_per_rnn 1 --gru_layers 1 --gen_per_sample 50 --grid 5 --phase $VAL_SET --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --max_dataset_size $VAL_SIZE --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 

