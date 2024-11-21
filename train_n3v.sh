GPU=1
PORT_BASE=6017
GT_PATH=/media/data3/code/lzh/original_data/dynerf

DATASET=dynerf
SAVE_PATH=/media/data3/code/lzh/4DGaussian/output

SCENE_LIST=(
    coffee_martini
    cook_spinach
    cut_roasted_beef
    # flame_salmon_frag1
    # flame_salmon_frag2
    # flame_salmon_frag3
    # flame_salmon_frag4
    flame_steak
    sear_steak
)
for SCENE in "${SCENE_LIST[@]}"; do
    echo "scene: $SCENE"
    CONFIG=$SCENE
    CUDA_VISIBLE_DEVICES=$GPU python train_Knnloss.py -s $GT_PATH/$SCENE --port $(expr $PORT_BASE + $GPU)  --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py
    CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG/  --skip_train --configs arguments/$DATASET/$CONFIG.py
    CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
done