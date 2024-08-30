import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
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

# TODO: remove this nonsense
import skrl
from skrl.utils import set_seed

import torch

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper, process_skrl_cfg


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
  env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
  set_seed(args_cli.seed if args_cli.seed is not None else agent_cfg["seed"])

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
  env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

  print(f'Env: {env}\n{dir(env)}')
  out = env.reset()
  print(f'out: {out}')
  print(f'out 0: {out[0]["policy"]}')
  print(f'\n{type(out[0]["policy"])} // {out[0]["policy"].shape}')


  action = env.action_space.sample()
  print(f'action: {action.dtype} // {action.shape}')
  next_ts = env.step(torch.Tensor(action))

  print(f'next_ts: {next_ts}')

  out = env.render()
  print(f'out: {out} // {out.sum()} // {out.shape}')

  env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
