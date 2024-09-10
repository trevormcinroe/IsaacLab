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
import sys

from omni.isaac.lab.app import AppLauncher
import numpy as np
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=9999999999, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--hw", type=int, default=None, help="hw of the env")
parser.add_argument("--latent_dim", type=int, default=50, help="z-dim")
parser.add_argument("--rollout_h", type=int, default=50, help="len of PPO rollout")
parser.add_argument('--run_notes', default=None, type=str, help='notes for the run')
parser.add_argument("--record", action="store_true", default=False, help="Record videos during eval.")
parser.add_argument('--learning_epochs', type=int, default=8)
parser.add_argument('--mini_batches', type=int, default=8)


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
parser.add_argument("--obs_type", type=str, action="store", default=1, help="gt, image")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime
import traceback

import carb
import skrl
from skrl.utils import set_seed

if args_cli.ml_framework.startswith("torch"):
    
    from skrl.memories.torch import RandomMemory
    # from skrl.trainers.torch import SequentialTrainer
    from multimodal_gym.utils.skrl.trainer import SequentialTrainer
    from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model
    # from skrl.utils.model_instantiators.torch.custom_models import custom_model
    from multimodal_gym.utils.skrl.models import custom_model, custom_deterministic_model, custom_gaussian_model
    from multimodal_gym.utils.frame_stack import FrameStack
    from multimodal_gym.utils.skrl.ppo import PPO, PPO_DEFAULT_CONFIG



from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

# from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# import the scheduler class
from torch.optim.lr_scheduler import StepLR, ExponentialLR

# Import extensions to set up environment tasks
import omni.isaac.lab_tasks  # noqa: F401
import multimodal_gym.tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from multimodal_gym.utils.skrl.skrl_wrapper import SkrlVecEnvWrapper, process_skrl_cfg
from omni.isaac.lab.envs import ViewerCfg


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.frame_stack = args_cli.frame_stack
    env_cfg.obs_type = args_cli.obs_type
    print("obs type:", env_cfg.obs_type)

    agent_cfg["agent"]["rollouts"] = args_cli.rollout_h
    agent_cfg["agent"]["learning_epochs"] = args_cli.learning_epochs
    agent_cfg["agent"]["mini_batches"] = args_cli.mini_batches

    set_seed(args_cli.seed if args_cli.seed is not None else agent_cfg["seed"])
    # multi-gpu training config
    if args_cli.distributed:
        if args_cli.ml_framework.startswith("jax"):
            raise ValueError("Multi-GPU distributed training not yet supported in JAX")
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env_cfg.hw = args_cli.hw
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Accessing the config this way may result in "CUDA error: an illegal memory access was encountered"
    # env.cfg.tiled_camera.height = args_cli.hw
    # env.cfg.tiled_camera.width = args_cli.hw

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # frame stack
    if args_cli.frame_stack != 1:
        env = FrameStack(env, num_stack=env_cfg.frame_stack)
        print(env.observation_space)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)#, obs_type=experiment_cfg["models"]["policy"]["obs_type"])  # same as: `wrap_env(env, wrapper="isaaclab")`

    print()

    print(env, env.observation_space)

    out, info = env.reset()

    print(f'out: {out.shape} [{out.min(), out.max()}]')

    # out = out.reshape(out.shape[0], -1)
    # out = out.reshape(64, 84, 84, 9)
    # # torch.Size([64, 84, 84, 9])
    # file_path = '/home/tmci/IsaacLab/IsaacLabExtension/exts/multimodal_gym/multimodal_gym/tasks/franka/lift.png'
    # import numpy as np
    # from PIL import Image
    # # .transpose(2, 0)
    # obs = np.array(out[10, :, :, :3].cpu() * 255).astype(np.uint8)
    # img = Image.fromarray(obs)
    # img.save(file_path)
    # # print(f'info: {info}')
    # qqq

    # instantiate models using skrl model instantiator utility
    print(f'agent_cfg: {agent_cfg} //\n{type(agent_cfg)}')
    agent_cfg["models"]["policy"]["img_dim"] = args_cli.hw
    agent_cfg["models"]["value"]["img_dim"] = args_cli.hw
    agent_cfg["models"]["policy"]["latent_dim"] = args_cli.latent_dim
    agent_cfg["models"]["value"]["latent_dim"] = args_cli.latent_dim

    models = {}
    # agent_cfg["models"]["policy"]["img_dim"] = args_cli.hw
    # agent_cfg["models"]["value"]["img_dim"] = args_cli.hw
    # non-shared models
    if agent_cfg["models"]["separate"]:
        models["policy"] = custom_gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(agent_cfg["models"]["policy"], ml_framework=args_cli.ml_framework),
            obs_type=env_cfg.obs_type,
            frame_stack=env_cfg.frame_stack,
            num_gt_observations=env_cfg.num_gt_observations,
        )
        models["value"] = custom_deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(agent_cfg["models"]["value"], ml_framework=args_cli.ml_framework),
            obs_type=env_cfg.obs_type,
            frame_stack=env_cfg.frame_stack,
            num_gt_observations=env_cfg.num_gt_observations
        )
    # shared models
    else:
        models["policy"] = shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(agent_cfg["models"]["policy"], ml_framework=args_cli.ml_framework),
                process_skrl_cfg(agent_cfg["models"]["value"], ml_framework=args_cli.ml_framework),
            ],
            frame_stack=env_cfg.frame_stack,
            num_gt_observations=env_cfg.num_gt_observations
        )
        models["value"] = models["policy"]

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    memory_size = agent_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # print(f'mem: {memory}')
    # qqq
    # action = models['value']({'states': out})
    # print(f'action: {action}')
    # qqq

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
    default_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    default_agent_cfg.update(process_skrl_cfg(agent_cfg["agent"], ml_framework=args_cli.ml_framework))

    default_agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    default_agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    # agent_cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * agent_cfg["rewards_shaper_scale"]

    # agent_cfg["learning_rate_scheduler"] = ExponentialLR
    # agent_cfg["learning_rate_scheduler_kwargs"] = {"gamma": 0.99}
        # update saving stuff from agent cfg

    # env_cfg.write_image_to_file = agent_cfg["logging"]["write_image_to_file"]
    # env_cfg.img_dir = agent_cfg["logging"]["img_dir"]

    agent = PPO(
        models=models,
        memory=memory,
        cfg=default_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    qqq

    # configure and instantiate a custom RL trainer for logging episode events
    trainer_cfg = agent_cfg["trainer"]
    trainer_cfg["close_environment_at_exit"] = False
    trainer_cfg["disable_progressbar"] = False
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # import torch
    # from multimodal_gym.utils.image_utils import save_images_to_file
    # for _ in range(5):
    #     action = torch.tensor([env.action_space.sample() for _ in range(args_cli.num_envs)])
    #     # print(f'action: {action}')
    #     next_obs, reward, term, trunc, info = env.step(action)
    # # # print(f'right at the end.')
    # # # print(f'o: {next_obs.shape}')
    # # # print(f'r: {reward}')
    # # # print(f'tt: {term} // {trunc}')
    # # # print(f'info: {info}')
    # # # img_dir = f"{__file__.replace('lift.py', '')}"
    # # # file_path = os.path.join(img_dir, f"{name}.png")
    # # make (N, H, W, C) for saving images
    # file_path = '/home/tmci/IsaacLab/IsaacLabExtension/exts/multimodal_gym/multimodal_gym/tasks/franka/lift.png'
    # save_images_to_file(
    #     next_obs.reshape(args_cli.num_envs, args_cli.hw, args_cli.hw, 3*args_cli.frame_stack)[:, :, :, :3],
    #     file_path)
    # qqq
    # train the agent
    import json
    import wandb
    with open('../../wandb.txt', 'r') as f:
        API_KEY = json.load(f)['api_key']

    os.environ['WANDB_API_KEY'] = API_KEY
    os.environ['WANDB_DIR'] = './wandb'
    os.environ['WANDB_CACHE_DIR'] = './wandb'
    os.environ['WANDB_CONFIG_DIR'] = './wandb'
    os.environ['WANDB_DATA_DIR'] = './wandb'

    config = {'rollout_h': args_cli.rollout_h, 'num_envs': args_cli.num_envs, 'run_notes': args_cli.run_notes}

    wandb.init(
        project='franka-lift',
        entity='trevor-mcinroe',
        name=f'{args_cli.run_notes}',
        config=config
    )

    # .reshape(64, 84, 84, 9)
    # torch.Size([64, 84, 84, 9])
    # print(out.shape)
    # frames = [out.reshape(args_cli.num_envs, args_cli.hw, args_cli.hw, 3 * args_cli.frame_stack)]
    # for _ in range(50):
    #     action = torch.tensor([env.action_space.sample() for _ in range(args_cli.num_envs)])
    #     # print(f'action: {action}')
    #     next_obs, reward, term, trunc, info = env.step(action)
    #     # print(f'next: {next_obs.shape}')
    #     # qqq
    #     frames.append(next_obs.reshape(args_cli.num_envs, args_cli.hw, args_cli.hw, 3 * args_cli.frame_stack))
    #
    # frames = torch.concat([x.unsqueeze(1) for x in frames], 1)[0].cpu()[:, :, :, :3].transpose(1, -1)
    # print(f'frames: {frames.shape}')
    # wandb.log({'video': wandb.Video((frames.numpy() * 255).astype(np.uint8), fps=30)})
    # qqq

    # Every call to .train() is 10k env steps
    for step in range(10_000_000 // (5_000 * args_cli.num_envs)):
        # Eval routine
        if args_cli.record:
            eval_returns, images = trainer.eval(True)
            # each is [63504]
            print(f'1st: {images[0].shape} // {len(images)}')

            frame_stack_divisor = (3 * args_cli.frame_stack) // 3
            adj_frames = []
            for frame in images:
                obs = frame.split(frame.shape[-1] // frame_stack_divisor, -1)
                obs = [
                    x.reshape(1, args_cli.hw, args_cli.hw, 3).transpose(-1, 1)
                    for x in obs
                ]
                # print(f'obs: {[x.shape for x in obs]}')
                obs = torch.cat(obs, 1)
                adj_frames.append(obs)

            adj_frames = torch.cat(adj_frames, 0).cpu()

            gen = np.array(adj_frames[:, :3] * 255).astype(np.uint8)
            wandb.log({'video': wandb.Video(gen, fps=30)})
            qqq
        else:
            eval_returns = trainer.eval()
        # print(f'Step {step * 5_000 * args_cli.num_envs}: {eval_returns.mean()}')
        logged_items = {k: v.mean().cpu() for k, v in eval_returns.items()}
        wandb.log({'global_steps': step * 5_000 * args_cli.num_envs, **logged_items})
        trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()