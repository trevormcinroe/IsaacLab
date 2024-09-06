# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents, shadow_hand_env_cfg

##
# Register Gym environments.
##

print("Registering shadow environments")

gym.register(
    id="Shadow",
    entry_point="multimodal_gym.tasks.shadow.inhand:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": shadow_hand_env_cfg.ShadowHandEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
