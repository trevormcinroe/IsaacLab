# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import os
import torch

from multimodal_gym.distractors.dots_source import DotsSource
from multimodal_gym.tasks.cartpole.base_cartpole import CartpoleEnv, CartpoleEnvCfg
from multimodal_gym.utils.image_utils import save_images_to_file

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.utils import configclass


@configclass
class RGBCartpoleEnvCfg(CartpoleEnvCfg):

    # camera
    # rgb tiled api only provides ambient rgb (they are working on lighting/shadows)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=20.0, replicate_physics=True)

    # occlusions
    occlusion = False
    write_image_to_file = False
    frame_stack = 1

    pole_rgb = (12,20,36)
    cart_rgb = (44,60,84)


@configclass
class DepthCartpoleEnvCfg(RGBCartpoleEnvCfg):
    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )

    # env
    num_channels = 1
    num_img_observations = num_channels * tiled_camera.height * tiled_camera.width


class CameraCartpoleEnv(CartpoleEnv):
    cfg: RGBCartpoleEnvCfg | DepthCartpoleEnvCfg

    def __init__(self, cfg: RGBCartpoleEnvCfg | DepthCartpoleEnvCfg, render_mode: str | None = None, **kwargs):
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
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_states = self.cfg.num_states
        self.num_channels = 3 * self.cfg.frame_stack
        self.num_img_observations = self.num_channels * self.cfg.tiled_camera.height * self.cfg.tiled_camera.width
        print("num channels:", self.num_channels)

        # set up spaces
        self.single_observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_img_observations,),
                )    

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)



    def _setup_scene(self):
        """Add the camera."""
        super()._setup_scene()
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

    def _get_observations(self) -> dict:
        obs = self._get_images()

        return obs

    def _get_images(self):
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"

        # returned data will be (num_cameras, height, width, num_channels) ((1024, 80, 80, 3)) torch.float32
        img_batch = self._tiled_camera.data.output[data_type].clone()
        batch_size = img_batch.size()[0]

        # make bg white
        img_batch[img_batch == 0] = 1

        # make half white
        # Calculate the midpoint of the height
        midpoint = 50

        # Make the top half of each image white
        img_batch[:, :midpoint, :, :] = 255

        if self.cfg.occlusion:
            # get updated disc masks
            disc_img, disc_mask = self.discs.get_image()
            disc_img = (disc_img.astype(np.float32)) / 255

            # mask the image batch
            disc_img_tensor = (
                torch.tensor(disc_img, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.sim.device)
            )
            disc_mask_tensor = (
                torch.tensor(disc_mask, dtype=torch.bool).unsqueeze(2).repeat(batch_size, 1, 1, 3).to(self.sim.device)
            )
            img_batch[disc_mask_tensor] = disc_img_tensor[disc_mask_tensor]

        if self.cfg.write_image_to_file:
            name = self.count
            name = "cart_only"
            img_dir = "/workspace/isaaclab/IsaacLabExtension/images"
            file_path = os.path.join(img_dir, f"{name}.png")
            save_images_to_file(img_batch, file_path)
            # cv2.imwrite(f'/home/epoch_numelle/Videos/isaac_lab/render/{self.count}_cv2.png', masked_img_batch[0])
            # self.count += 1

        flattened_images = img_batch.view(batch_size, -1)

        return flattened_images
