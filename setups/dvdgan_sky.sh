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
DATASETS_DIR=/mnt/raid/patrickradner/datasets/
CHECKPOINT_DIR=/mnt/raid/patrickradner/checkpoints
RESULTS_DIR=/mnt/raid/patrickradner/results


DATASET=sky_timelapse
DATASET_MODE=sky

# models
MODEL=dvdgan
# optimizer parameters
LR=0.0003
BATCHSIZE=128
SUB_BATCH=1
RESOLUTION=128
FPS=25
GENERATOR=dvdgan

#NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_noise_cgan_aux_seg
NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}_base
git add -A 
git commit -m $NAME
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1.0

FREQ=200
DISP_FRAMES=16
VAL_SIZE=100
VAL_SET=test #split
#--verbose --sanity_check
python train.py --ch_g 32 --ch_ds 16 --ch_dt 64 --no_dt_prepool --conditional --start_fp 2 --max_fp 5 --up_blocks_per_rnn 1 --lr_policy cosine --niter 10 --niter_decay 250 --clip_grads 1 --n_critic 3 --lambda_L1 50 --lambda_S 1 --lambda_T 5 --lambda_AUX 0 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set $VAL_SET --pretrain_epochs 0 --tlg .3 --tld .7 --max_val_dataset_size $VAL_SIZE --init_type orthogonal --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
python test.py --ch_g 32 --ch_ds 16 --ch_dt 64 --no_dt_prepool --conditional --start_fp 2 --max_fp 5 --up_blocks_per_rnn 1 --grid 1 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 

DATASET=gaugan
DATASET_MODE=image
#python test.py --grid 5 --phase val --dataroot $DATASETS_DIR/$DATASET --results_dir $RESULTS_DIR --max_clip_length $LEN --skip_frames $SKIP --batch_size $BATCHSIZE --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID 
