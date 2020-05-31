set -ex

DATASETS_DIR=/mnt/raid/patrickradner/datasets


 #renderer
#RENDERER=MultiTarget-neuralRenderer_200
RENDERER=no_renderer

 #optimizer parameters
LR=0.001
shCH_SIZE=1

 #GPU
GPU_ID="0"

 #display params
DISP_FREQ=50
LOSS=L1

EPOCH=latest

#source "./experiment_setups/PerPixel_lab3_vgg.sh"
#source "./experiment_setups/PerPixel2_lab3_vgg.sh"
source "./experiment_setups/PerPixel2_lab3_vgg_small.sh"
#source "./experiment_setups/PerPixel2_lab3_gan_small.sh"
#source "./experiment_setups/PerPixel_lab3.sh"
#source "./experiment_setups/GruPerPixel_lab3.sh"
#source "./experiment_setups/Blend_lab3.sh"
#source "./experiment_setups/LstmPerPixel_4_4_lab3.sh"
#source "./experiment_setups/Lstm2UNET3_lab2.sh"
#source "./experiment_setups/PerLayerPerPixel4_lab2.sh"
#source "./experiment_setups/LstmPerPixel4_lab_2.sh"
#source "./experiment_setups/PerPixel4_lab_2.sh"
#source "./experiment_setups/UNET_5_lab_2.sh"
#source "./experiment_setups/Blend_lab_2.sh"
#source "./experiment_setups/Debug.sh"


if [[ $(nvidia-smi | grep "^|    $GPU_ID    ") ]]; then
    read -p "GPU currently in use, continue? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]
    then
        exit 1
    fi
fi

python test.py --rendererType $RENDERER_TYPE --num_depth_layers $NUM_DEPTH_LAYERS --name $NAME --epoch $EPOCH --display_winsize 512 --tex_dim $TEX_DIM --tex_features $TEX_FEATURES --dataroot $DATASETS_DIR/$DATA  --lossType $LOSS --model $MODEL --netG unet_256 --dataset_mode $DATASET_MODE --norm batch --gpu_ids $GPU_ID $OPTIONS


