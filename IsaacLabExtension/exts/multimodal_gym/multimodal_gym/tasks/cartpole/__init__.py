# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

# from .camera_cartpole import DepthCartpoleEnvCfg, RGBCartpoleEnvCfg
# from .prop_cartpole import PropCartpoleEnvCfg
# from .multimodal_cartpole import MultimodalCartpoleEnvCfg
from .cartpole import CartpoleEnvCfg

##
# Register Gym environments.
##

print("Registering cartpole environments")


gym.register(
    id="PropCartpole",
    entry_point="multimodal_gym.tasks.cartpole.cartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_prop_ppo.yaml",
    },
)


gym.register(
    id="CameraCartpole",
    entry_point="multimodal_gym.tasks.cartpole.cartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo.yaml",
    },
)


gym.register(
    id="ConcatCartpole",
    entry_point="multimodal_gym.tasks.cartpole.cartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_concat_ppo.yaml",
    },
)


# gym.register(
#     id="DepthCartpole",
#     entry_point="multimodal_gym.tasks.cartpole.camera_cartpole:DepthCartpoleEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": DepthCartpoleEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#     },
# )
