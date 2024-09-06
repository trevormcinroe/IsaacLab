#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my_isaac
cd /home/elle/code/external/IsaacLab/IsaacLabExtension

PYTHON_EXE=/home/elle/miniconda3/envs/my_isaac/bin/python
# List of seeds
SEEDS=(42 123 456 789 101112)

# for SEED in "${SEEDS[@]}"; do
#     echo "Running with seed $SEED"
#     $PYTHON_EXE scripts/rl_games/train.py --task MultiCartpole --headless --enable_cameras --seed "$SEED"
# done

NUM_DISCS=5 #(1 5 10)
DOT_BEHAVIOURS=("linear" "episode" "random")
SEEDS=(42 123 456)

for DOT_BEHAVIOUR in "${DOT_BEHAVIOURS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running with seed $SEED"
        EXP_NAME="$NUM_DISCS-$DOT_BEHAVIOUR-discs-$SEEDS-seed"
        $PYTHON_EXE scripts/rl_games/train.py --task MultiCartpole --headless --enable_cameras --seed "$SEED" --occlusion --dots_behaviour "$DOT_BEHAVIOUR" --num_discs "$NUM_DISCS" --exp_name "$EXP_NAME"
    done
done
