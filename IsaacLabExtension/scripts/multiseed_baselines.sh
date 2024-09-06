#!/bin/bash

# eval "$(conda shell.bash hook)"
# conda activate my_isaac
# cd /home/elle/code/external/IsaacLab/IsaacLabExtension

PYTHON_EXE=/workspace/isaaclab/_isaac_sim/python.sh #/home/elle/miniconda3/envs/my_isaac/bin/python
# List of seeds
SEEDS=(42 123 456)
OBS_TYPE=("PropCartpole" "CameraCartpole" "MultiCartpole")

for OBS in "${OBS_TYPE[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running with seed $SEED"
        if [ "$OBS" == "PropCartpole" ]; then
            $PYTHON_EXE scripts/skrl/train.py --task $OBS --headless --seed "$SEED"
        else
            $PYTHON_EXE scripts/skrl/train.py --task $OBS --headless --seed "$SEED" --enable_cameras --frame_stack 3
        fi
    done
done
