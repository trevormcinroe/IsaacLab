# Multimodal Gym

## Todo

### Franka Lift

General training
- [ ] Fix training from pixels
- [ ] Put in contact sensors
- [ ] Have configurations to train with all combinations

Environment improvements
- [ ] Put block in box
- [ ] Make block slippery
- [ ] Make block deformable
- [ ] Make block brittle

Random nice stuff
- [ ] reward curriculum
- [ ] place goal command
- [ ] cabinet


objects
011_banana
019_pitcher_base
037_scissors
040_large_marker
small_KLT
torus
tube




## Running 
```
python scripts/skrl/train.py --task FrankaLift --headless --obs_type gt
python scripts/skrl/train.py --task ImageFrankaLift --headless --enable_cameras --frame_stack 3 --num_envs 32 --obs_type image

python scripts/skrl/train.py --task Shadow --headless --obs_type full
python scripts/skrl/train.py --task Shadow --headless --enable_cameras --frame_stack 1 --num_envs 1 --obs_type image
```

## Installation
### IsaacLab
IsaacLab installation instructions:
```
git@github.com:elle-miller/IsaacLab.git
cd IsaacLab
```
Install with binaries
```
ln -s /home/elle/.local/share/ov/pkg/isaac-sim-4.0.0 _isaac_sim
./isaaclab.sh --conda isaac
conda activate isaac
```

Install via pip
```
conda create -n isaaclab python=3.10
conda activate isaaclab
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
```
Install and verify:
```
./isaaclab.sh --install
pip install scipy boto3
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Reach-Franka-v0 --headless
```

Install my extension (conda)
```
git clone git@github.com:elle-miller/IsaacLabExtension.git
cd IsaacLabExtension/exts/multimodal_gym

# if isaaclab -p exists
isaaclab -p -m pip install -e .
cd ../..
python scripts/skrl/train.py --task PropCartpole --headless
# else (pip installation)
python -m pip install -e .
cd ../..
python scripts/skrl/train.py --task FrankaLift --headless

```

### Local `rl_games`
Install my version
```
git clone git@github.com:elle-miller/rl_games.git
cd rl_games
```
Uninstall normal `rl_games`
```
isaaclab -p -m pip uninstall rl_games
isaaclab -p -m pip install -e .
isaaclab -p -m pip install ray
isaaclab -p -m pip list # verify local dir
```

## Using


```
python scripts/skrl/train.py --task CameraFrankaLift --headless --num_envs 64 --enable_cameras --frame_stack 3
python scripts/skrl/train.py --task CameraFrankaLift --headless --num_envs 1 --enable_cameras 


isaaclab -p scripts/skrl/train.py --task FrankaLift --headless
isaaclab -p scripts/skrl/play.py --task FrankaLift --num_envs 4

isaaclab -p scripts/skrl/train.py --task Shadow --headless

isaaclab -p scripts/skrl/train.py --task PropCartpole --headless
isaaclab -p scripts/skrl/train.py --task CameraCartpole --headless --enable_cameras --frame_stack 3 --seed 42
isaaclab -p scripts/skrl/sweep.py --task CameraCartpole --headless --enable_cameras --frame_stack 3 --seed 42

isaaclab -p scripts/skrl/train.py --task MultiCartpole --headless --enable_cameras --frame_stack 3 --seed 42


isaaclab -p scripts/rl_games/train.py --task PropCartpole --headless
isaaclab -p scripts/rl_games/train.py --task CameraCartpole --headless --enable_cameras --frame_stack 3 --seed 42
isaaclab -p scripts/rl_games/train.py --task MultiCartpole --headless --enable_cameras --frame_stack 3 --seed 42
``` 

on cluster
```
isaaclab -p scripts/rl_games/train.py --task CameraCartpole --headless --enable_cameras --num_envs 2048
```

### Hyperparameter optimisation
To run a hyperparameter optimisation sweep, create the config file. Make sure wandb is disabled.
```
isaaclab -p scripts/skrl/sweep.py --task PropCartpole --headless --num_envs 2048
optuna-dashboard sqlite:///hyperparameter_optimization.db 
```

### occlusions
```
isaaclab -p scripts/rl_games/train.py --task MultiCartpole --headless --enable_cameras --occlusion --dots_behaviour linear --num_discs 5
```


### Taking videos
To render training: `--video`

```
isaaclab -p scripts/rl_games/train.py --task CameraCartpole --headless --enable_cameras --frame_stack 3 --seed 42 --occlusion --dots_behaviour episode --num_discs 10 --video --video_interval 100000
```

### Play trained agent

Without mentioning a checkpoint, best one will be played:
```
isaaclab -p scripts/rl_games/play.py --task PropCartpole --num_envs 1
```
### Plotting
```
python exts/multimodal_gym/multimodal_gym/utils/plot.py
```

## Docker

change `docker-compose.yaml`
```
x-default-isaac-lab-deploy: &default-isaac-lab-deploy
  resources:
    limits:
      cpus: "16"              # number of cores
      memory: 64g           # limit container usage
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["7"]  # specify the GPU ID here if necessary
          capabilities: [ gpu ]
```

```
  - type: bind
    source: ../IsaacLabExtension
    target: ${DOCKER_ISAACLAB_PATH}/IsaacLabExtension
  - type: bind
    source: ../rl_games
    target: ${DOCKER_ISAACLAB_PATH}/rl_games
  - type: bind
    source: ../skrl
    target: ${DOCKER_ISAACLAB_PATH}/skrl
```


add following lines to `Dockerfile.base`
```
# install extension
WORKDIR ${ISAACLAB_PATH}/IsaacLabExtension/exts/multimodal_gym
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .

# skrl stuff
WORKDIR ${ISAACLAB_PATH}/skrl
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y skrl
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .["torch"]
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install optuna optuna-dashboard

# install local rl games
WORKDIR ${ISAACLAB_PATH}/rl_games
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y rl_games
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install ray

# set workdir
WORKDIR ${ISAACLAB_PATH}/IsaacLabExtension
```
Then from `IsaacLab` run:
```
./docker/container.sh start
./docker/container.sh enter
cd IsaacLabExtensionTemplate
isaaclab -p scripts/rl_games/train.py --task Elle-Cartpole-Direct-v0 --headless
```
when finished
```
./docker/container.sh copy

docker cp isaac-lab-base:/workspace/isaaclab/IsaacLabExtension/logs /home/emil/code/external/IsaacLab/IsaacLabExtension/logs
```

## Code formatting

We have a pre-commit template to automatically format your code.

```bash
pre-commit run --all-files
```

# vscode
set conda interpreter
in /code/.vscode/launch.json
```
{
            "name": "Python: Train Environment",
            "type": "debugpy",
            "request": "launch",
            "args" : ["--task", "CameraCartpole", "--headless", "--enable_cameras"],
            "program": "${workspaceFolder}/external/IsaacLab/IsaacLabExtension/scripts/rl_games/train.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.vscode/.python.env",
        },
```


## Networks

### Proprioception
```
# shared net
Sequential(
  (0): Linear(in_features=4, out_features=32, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=32, out_features=32, bias=True)
  (3): ELU(alpha=1.0)
)
# mean net
Sequential(
  (0): Linear(in_features=32, out_features=1, bias=True)
  (1): Tanh()
)
# value net
Sequential(
  (0): Linear(in_features=32, out_features=1, bias=True)
  (1): Identity()
)
```

### Camera
```
Sequential(
  (0): ImageEncoder(
    (convs): ModuleList(
      (0): Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2))
      (1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    )
    (head): Sequential(
      (0): Linear(in_features=43808, out_features=50, bias=True)
      (1): LayerNorm((50,), eps=1e-05, elementwise_affine=True)
    )
  )
  (1): Linear(in_features=50, out_features=32, bias=True)
  (2): ELU(alpha=1.0)
  (3): Linear(in_features=32, out_features=32, bias=True)
  (4): ELU(alpha=1.0)
)
```

- compare architecture with RL Games
  - num features, num layers, kernel, stride
  
- is frame stacking actually working 



### Isaac Lab documentation that already should exist

Rigid object
```
# X of all bodies in simulation world frame
body_acc_w
body_ang_vel_w
body_lin_vel_w
body_pos_w
body_quat_w
body_vel_w

# root X in simulation world frame
root_pos_w  
root_quat_w (w, x, y, z)
root_ang_vel_w
root_lin_vel_w
root_state_w [pos, quat, lin_vel, ang_vel]
root_vel_w [lin_vel, ang_vel]

# root X in base frame
root_ang_vel_b
root_lin_vel_b
```

