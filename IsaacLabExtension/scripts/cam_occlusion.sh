#!/bin/bash

# eval "$(conda shell.bash hook)"
# conda activate my_isaac
# cd /home/elle/code/external/IsaacLab/IsaacLabExtension

# PYTHON_EXE=/home/elle/miniconda3/envs/my_isaac/bin/python
PYTHON_EXE=/workspace/isaaclab/_isaac_sim/python.sh

# List of seeds
SEEDS=(47 789 101112)

NUM_DISCS=(5 10 15)
# DOT_BEHAVIOURS=("linear" "episode" "random")
DOT_BEHAVIOUR=("random")
NUM_DISCS=(5 10 15)
for SEED in "${SEEDS[@]}"; do
    for NUM_DISC in "${NUM_DISCS[@]}"; do 
        echo "Running with seed $SEED"
        EXP_NAME="CameraCartpole-$DOT_BEHAVIOUR-$NUM_DISC"
        $PYTHON_EXE scripts/rl_games/train.py --task CameraCartpole --exp_name "$EXP_NAME" --headless --enable_cameras --frame_stack 3 --seed "$SEED" --occlusion --dots_behaviour "$DOT_BEHAVIOUR" --num_discs "$NUM_DISC"
    done
done

# concat
# DOT_BEHAVIOURS=("linear" "episode" "random")
# for DOT_BEHAVIOUR in "${DOT_BEHAVIOURS[@]}"; do
#     for NUM_DISC in "${NUM_DISCS[@]}"; do
#         for SEED in "${SEEDS[@]}"; do
#             echo "Running with seed $SEED"
#             EXP_NAME="concat-$NUM_DISC-$DOT_BEHAVIOUR-discs"
#             $PYTHON_EXE scripts/rl_games/train.py --task MultiCartpole --headless --enable_cameras --seed "$SEED" --occlusion --dots_behaviour "$DOT_BEHAVIOUR" --num_discs "$NUM_DISC" --exp_name "$EXP_NAME"
#         done
#     done
# done