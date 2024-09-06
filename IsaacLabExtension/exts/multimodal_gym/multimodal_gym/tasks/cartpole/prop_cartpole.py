# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from multimodal_gym.tasks.cartpole.base_cartpole import CartpoleEnv, CartpoleEnvCfg

from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass


@configclass
class PropCartpoleEnvCfg(CartpoleEnvCfg):
    # needed in DirectRL setup
    num_observations = 2
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)


class PropCartpoleEnv(CartpoleEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_observations(self) -> dict:
        obs = self._get_proprioception()
        return obs
