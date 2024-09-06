# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument("--frame_stack", type=int, action="store", default=1, help="Choose from static, moving")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime
import time
import traceback

import carb
import skrl
from skrl.utils import set_seed
from skrl.trainers.torch import StepTrainer

import optuna
import torch
import logging
import numpy as np

# disable skrl logging
from skrl import logger
logger.setLevel(logging.WARNING)

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model
from multimodal_gym.utils.skrl.models import custom_model
from multimodal_gym.utils.frame_stack import FrameStack
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
# from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg
from multimodal_gym.utils.skrl.skrl_wrapper import SkrlVecEnvWrapper, process_skrl_cfg

from multimodal_gym.utils.misc import to_numpy, to_cpu
# Import extensions to set up environment tasks
import multimodal_gym.tasks  # noqa: F401


study_name = "CameraCartpole-SKRL"


# parse configuration
env_cfg = parse_env_cfg(
    args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
env_cfg.frame_stack = args_cli.frame_stack
# create isaac environment
print("Creating isaaclab environment...")
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
print(env.observation_space)

# frame stack
if args_cli.frame_stack != 1:
    env = FrameStack(env, num_stack=env_cfg.frame_stack)
    print(env.observation_space)
# wrap around environment for skrl
env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="isaaclab")`


import gc
import torch

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

def objective(trial: optuna.Trial):

    print(f"Starting trial: {trial.number}")

    # read the seed from command line
    args_cli_seed = args_cli.seed

    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    """Train with skrl agent."""

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        experiment_cfg["trainer"]["timesteps"] = args_cli.max_iterations * experiment_cfg["agent"]["rollouts"]

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)


    # frame stack
    # env = gym.wrappers.FrameStack(env, 4)
    # print(env.observation_space)

    # set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html
    models = {}
    if not experiment_cfg["models"]["separate"]:
        models["policy"] = custom_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(experiment_cfg["models"]["policy"], ml_framework=args_cli.ml_framework),
                process_skrl_cfg(experiment_cfg["models"]["value"], ml_framework=args_cli.ml_framework),
            ],
        )
        models["value"] = models["policy"]
    else:
        raise NotImplementedError

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # configure and instantiate PPO agent
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"], ml_framework=args_cli.ml_framework))
    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    # parameters to optimize
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    mini_batches = trial.suggest_categorical("mini_batches", [4, 8])
    learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=1e-3, log=True)
    learning_epochs = trial.suggest_categorical("learning_epochs", [4, 8, 16])
    rollouts = trial.suggest_categorical("rollouts", [8, 16])
    rewards_shaper_scale = trial.suggest_categorical("rewards_shaper_scale", [0.01, 0.1, 1])

    agent_cfg["rollouts"] = rollouts
    agent_cfg["mini_batches"] = mini_batches
    agent_cfg["learning_epochs"] = learning_epochs
    agent_cfg["learning_rate"] = learning_rate
    agent_cfg["rewards_shaper_scale"] = rewards_shaper_scale
    agent_cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * rewards_shaper_scale

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/api/trainers.html
    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["close_environment_at_exit"] = False
    trainer_cfg["disable_progressbar"] = False
    trainer = StepTrainer(cfg=trainer_cfg, env=env, agents=agent)

    #  train the agent
    trainer = StepTrainer(cfg=trainer_cfg, env=env, agents=agent)
    env.reset()
    mean_returns = 0
    highest_mean_returns = 0
    report_freq = 100
    for timestep in range(trainer_cfg["timesteps"]):
        trainer.train(timestep=timestep)
        # returns is a deque of upto 100 most recent episode returns
        returns = np.array(trainer.agents._track_rewards)

        # only report nonzero-length returns
        if len(returns) > 0:
            mean_returns =  np.mean(returns)
            if mean_returns > highest_mean_returns:
                highest_mean_returns = mean_returns

        # report
        if timestep % report_freq == 0:
            # print(timestep, mean_returns)
            trial.report(mean_returns, timestep)
            if trial.should_prune():
                free_memory()
                raise optuna.TrialPruned()
            
    free_memory()
    
    return highest_mean_returns


if __name__ == "__main__":
    try:
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
        storage = "sqlite:///hyperparameter_optimization.db"
        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.HyperbandPruner()
        direction = "maximize"  # maximize episode reward

        study = optuna.create_study(storage=storage,
                                    sampler=sampler,
                                    pruner=pruner,
                                    study_name=study_name,
                                    direction=direction,
                                    load_if_exists=True)
        

        study.optimize(objective, n_trials=20, show_progress_bar=True)

        print('Number of finished trials: ', len(study.trials))

        print(f"The best trial obtains a normalized score of {study.best_trial.value}", study.best_trial.params)

        ## at the very end
        env.close()
            # run the main function
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()


