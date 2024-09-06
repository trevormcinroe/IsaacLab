#!/bin/bash

# eval "$(conda shell.bash hook)"
# conda activate my_isaac
# cd /home/elle/code/external/IsaacLab/IsaacLabExtension


PYTHON_EXE=/workspace/isaaclab/_isaac_sim/python.sh #/home/elle/miniconda3/envs/my_isaac/bin/python
# List of seeds
SEEDS=(42 123 456 789 101112)

for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED"
    $PYTHON_EXE scripts/rl_games/train.py --task PropCartpole --headless --seed "$SEED"
done
