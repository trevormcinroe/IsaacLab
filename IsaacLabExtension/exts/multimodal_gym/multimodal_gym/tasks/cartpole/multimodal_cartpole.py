# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import os

from multimodal_gym.distractors.dots_source import DotsSource
from multimodal_gym.tasks.cartpole.base_cartpole import CartpoleEnv, CartpoleEnvCfg
from multimodal_gym.tasks.cartpole.camera_cartpole import CameraCartpoleEnv, RGBCartpoleEnvCfg
from multimodal_gym.utils.image_utils import save_images_to_file

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.utils import configclass

from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete


@configclass
class MultimodalCartpoleEnvCfg(RGBCartpoleEnvCfg):

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=20.0, replicate_physics=True)

    # occlusions
    # occlusion = False
    num_gt_observations = 2


class MultimodalCartpoleEnv(CameraCartpoleEnv):
    cfg: MultimodalCartpoleEnvCfg

    def __init__(self, cfg: MultimodalCartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

        if self.cfg.occlusion:
            self.discs = DotsSource(
                seed=42, shape=(80, 80), dots_behaviour=self.cfg.dots_behaviour, num_dots=self.cfg.num_discs
            )

        # if self.cfg.write_image_to_file and not os.path.exists(self.cfg.img_dir):
        #     os.mkdir(self.cfg.img_dir)
        
        self.count = 0


    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        """Update only observation spaces."""
        self.num_observations = int(self.num_img_observations) + int(self.cfg.num_gt_observations)

        # set up spaces
        self.single_observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_observations,),
                )         
        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)


    def _get_observations(self) -> dict:

        # proprioception
        proprioception = self._get_proprioception()

        flattened_images = self._get_images()

        # concatenate
        obs = torch.cat((proprioception, flattened_images), -1)

        # if self.cfg.write_image_to_file:
        #     file_path = os.path.join(self.cfg.img_dir, f"{self.count}.png")
        #     save_images_to_file(images, file_path)
        #     # cv2.imwrite(f'/home/epoch_numelle/Videos/isaac_lab/render/{self.count}_cv2.png', masked_img_batch[0])
        #     self.count += 1

        return obs
