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


#DATASET=movingmnist
#DATASET=3hr
# DATASET=cifar10
# DATASET_MODE=dummy
DATASET=river_relaxing
DATASET_MODE=video


# models
MODEL=lhc
# optimizer parameters
LR=1
BATCHSIZE=4
SUB_BATCH=4
RESOLUTION=64
FPS=25
GENERATOR=lhc

NAME=${DATASET}_${MODEL}_${GENERATOR}_${RESOLUTION}
DISPNAME=${NAME}
# vid settings
SKIP=1
LEN=1.0

FREQ=10
DISP_FRAMES=16
#--verbose --sanity_check
python train.py --verbose --max_dataset_size 1 --niter 250 --niter_decay 250 --train_mode "mixed" --clip_grads 500 --n_critic 1 --lambda_L1 1 --lambda_S 0 --lambda_T 0 --max_clip_length $LEN --skip_frames $SKIP --validation_freq 5 --validation_set val --pretrain_epochs 100 --tlg .2 --tld .8 --max_val_dataset_size 500 --init_type xavier --batch_size $BATCHSIZE --parallell_batch_size $SUB_BATCH  --resolution $RESOLUTION --fps $FPS --dataroot $DATASETS_DIR/$DATASET --checkpoints_dir $CHECKPOINT_DIR --name $NAME --model $MODEL --generator $GENERATOR --dataset_mode $DATASET_MODE --gpu_ids $GPU_ID --lr $LR --display_port $VISDOM_PORT --display_env $DISPNAME --update_html_freq $FREQ --num_display_frames $DISP_FRAMES --print_freq $FREQ --display_freq $FREQ
#python test.py --dataroot $DATASETS_DIR/$OBJECT --name $OBJECT-$MODEL --renderer $RENDERER --model $MODEL --netG unet_256 --dataset_mode aligned --norm batch --gpu_ids 0
