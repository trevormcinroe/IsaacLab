# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, lift, deformable_lift, occluded_lift, reach, manager_lift #ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##
print("Registering franka environments")

gym.register(
    id="FrankaLift",
    entry_point="multimodal_gym.tasks.franka.lift:LiftEnv",
    kwargs={
        "env_cfg_entry_point": lift.LiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="DeformableFrankaLift",
    entry_point="multimodal_gym.tasks.franka.deformable_lift:DeformableLiftEnv",
    kwargs={
        "env_cfg_entry_point": deformable_lift.DeformableLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="OccludedFrankaLift",
    entry_point="multimodal_gym.tasks.franka.occluded_lift:OccludedLiftEnv",
    kwargs={
        "env_cfg_entry_point": occluded_lift.OccludedLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="ImageFrankaLift",
    entry_point="multimodal_gym.tasks.franka.lift:LiftEnv",
    kwargs={
        "env_cfg_entry_point": lift.LiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_image_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="ImagePropFrankaLift",
    entry_point="multimodal_gym.tasks.franka.lift:LiftEnv",
    kwargs={
        "env_cfg_entry_point": lift.LiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_image_prop_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="FrankaReach",
    entry_point="multimodal_gym.tasks.franka.reach:ReachEnv",
    kwargs={
        "env_cfg_entry_point": reach.ReachEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="FrankaLiftManager",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": manager_lift.FrankaCubeLiftEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)