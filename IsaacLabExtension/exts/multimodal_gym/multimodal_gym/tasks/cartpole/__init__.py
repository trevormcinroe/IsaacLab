# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .camera_cartpole import DepthCartpoleEnvCfg, RGBCartpoleEnvCfg
from .prop_cartpole import PropCartpoleEnvCfg
from .multimodal_cartpole import MultimodalCartpoleEnvCfg

##
# Register Gym environments.
##

print("Registering cartpole environments")


gym.register(
    id="PropCartpole",
    entry_point="multimodal_gym.tasks.cartpole.prop_cartpole:PropCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PropCartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rlgames_prop_ppo.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_prop_ppo.yaml",
    },
)


gym.register(
    id="CameraCartpole",
    entry_point="multimodal_gym.tasks.cartpole.camera_cartpole:CameraCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RGBCartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rlgames_camera_ppo.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo.yaml",

    },
)

gym.register(
    id="DepthCartpole",
    entry_point="multimodal_gym.tasks.cartpole.camera_cartpole:DepthCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DepthCartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="MultiCartpole",
    entry_point="multimodal_gym.tasks.cartpole.multimodal_cartpole:MultimodalCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MultimodalCartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rlgames_multimodal_ppo.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_concat_ppo.yaml",
    },
)