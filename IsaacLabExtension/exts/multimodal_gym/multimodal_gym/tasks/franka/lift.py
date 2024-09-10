# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from collections.abc import Sequence
import os
from multimodal_gym.utils.isaaclab.direct_rl_env import DirectRLElleEnv
import gymnasium as gym
from omni.isaac.lab.assets import DeformableObjectCfg, DeformableObject

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers import VisualizationMarkersCfg


from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import FrameTransformerData

from multimodal_gym.utils.image_utils import save_images_to_file


"""
    self.sim.physx.bounce_threshold_velocity = 0.2
    self.sim.physx.bounce_threshold_velocity = 0.01
    self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
    self.sim.physx.friction_correlation_distance = 0.00625


obs_type

gt
image
prop_tactile
image_prop
image_tactile
image_prop_tactile

"""

@configclass
class LiftEnvCfg(DirectRLEnvCfg):
    # physics sim
    physics_dt = 0.01 # 100 Hz
    # number of physics step per control step
    decimation = 2
    # the number of physics simulation steps per rendering steps (default=1)
    render_interval = 2
    episode_length_s = 5.0  # 5 * 120 / 2 = 300 timesteps

    num_observations = 0
    num_actions = 9
    num_states = 0

    obs_type = "image"
    num_gt_observations = 38
    num_tactile_observations = 2
    num_prop_observations = 9 + 9 + 3   # joint pos, joint vel, ee pos
    

    reset_position_noise = 0.01

    # lift stuff
    minimal_height = 0.04
    reaching_object_scale = 1
    lift_object_scale = 15.0
    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0
    action_penalty_scale = -0.01
    joint_vel_penalty_scale = -0.01
    curriculum = False
    fall_dist = 1.0

    # target
    target_x_max = 0.6
    target_x_min = 0.4
    target_y_max = 0.25
    target_y_min = -0.25
    target_z_max = 0.5
    target_z_min = 0.25

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=20, replicate_physics=True)
    # lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viewer: ViewerCfg = ViewerCfg(eye=(1.75, 1.75, 3.0), lookat=(0.0, 0.0, 1.5))

    # robot
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # activate_contact_sensors=False,

    actuated_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2"
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1,0,0,0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
    
    table_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            )
        )
    
    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"
    ee_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )

    # perception
    # pos=(x, y, z) -- z [down, 0, up]
    # eye = [0, 1.5, 0.5]
    # target = [0.3, 0, 0.5]
    # perception
    # object pos=[0.5, 0, 0.055]
    # x [down-left, 0, up-right]
    # y [left, 0, right]
    # offset=TiledCameraCfg.OffsetCfg(pos=(-0.33, -0.35, 0.7), rot=(1,0,0,0), convention="world"),
    # in above, camera is either behind-right(facing for both) of arm
    # behind-left of arm pos=(-0.13, 0.20, 0.7) rot=(1,0,0,0) focal_length=3
    # side-right TiledCameraCfg.OffsetCfg(pos=(0.65, -0.60, 0.9), rot=(1,0,1,1), convention="world"), 15, y[more neg down-right]
    #  x [more pos up right]
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.85, -1.10, 1.6), rot=(1,0,1,1), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 3.2)
    #     ),
    #     width=84,
    #     height=84,
    #     #debug_vis=True
    # )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.85, -1.10, 1.6), rot=(1, 0, 1, 1), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 3.2)
        ),
        width=84,
        height=84,
        # debug_vis=True
    )

    write_image_to_file = True
    frame_stack = 1
    eye = [0.0, 0.0, 1.0] # orig: [0, 1.5, 0.5]
    target = [0.0, 0.0, 0.5]  # orig: [0.3, 0.0, 0.5]

class LiftEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: LiftEnvCfg

    def __init__(self, cfg: LiftEnvCfg, render_mode: str | None = None, **kwargs):
        # The **only** way to edit the hw of the camera is to do it **before** the parent class is init'ed.
        cfg.tiled_camera.height = cfg.hw
        cfg.tiled_camera.width = cfg.hw

        super().__init__(cfg, render_mode, **kwargs)

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # create empty tensors
        self.joint_pos = torch.zeros((self.num_envs, 9), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, 9), device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ee_distance = torch.zeros((self.num_envs,), device=self.device)
        self.object_goal_distance = torch.zeros((self.num_envs,), device=self.device)

        # save reward weights so they can be adjusted online
        self.reaching_object_scale = cfg.reaching_object_scale
        self.lift_object_scale = cfg.lift_object_scale
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.object_goal_tracking_finegrained_scale = cfg.object_goal_tracking_finegrained_scale
        self.action_penalty_scale = cfg.action_penalty_scale
        self.joint_vel_penalty_scale = cfg.joint_vel_penalty_scale

        # default goal positions
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.4, 0.2, 0.3], device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # camera stuff
        self.count = 0
        self.init = False      

    def get_num_observations(self):
        self.num_img_observations = self.cfg.frame_stack * 3 * self.cfg.tiled_camera.height * self.cfg.tiled_camera.width
        match self.cfg.obs_type:
            case "gt":
                return self.cfg.num_gt_observations
            case "image":
                return self.num_img_observations
            case "image_prop":
                return self.num_img_observations + self.cfg.num_prop_observations

        raise ValueError("Value doesn't match")

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.get_num_observations()
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations,))
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        # Issues with this particular rigid object
        # self.table = RigidObject(self.cfg.table_cfg)

        # self.box = RigidObject(self.cfg.box_cfg)

        # FrameTransformer provides interface for reporting the transform of
        # one or more frames (target frames) wrt to another frame (source frame)
        self.ee_frame = FrameTransformer(self.cfg.ee_config)
        self.ee_frame.set_debug_vis(True)
        # self.goal_frame = FrameTransformer(self.cfg.goal_object_cfg)
        # self.goal_frame.set_debug_vis(True)
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)

        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        # self.scene.rigid_objects["table"] = self.table

        self.scene.sensors["ee_frame"] = self.ee_frame

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.cfg.obs_type == "image" or self.cfg.obs_type == "image_prop":
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            # self._tiled_camera._initialize_impl()

            # eyes = torch.tensor(self.cfg.eye, dtype=torch.float, device=self.device).repeat(
            #     (self.num_envs, 1)) + self.scene.env_origins
            # targets = torch.tensor(self.cfg.target, dtype=torch.float, device=self.device).repeat(
            #     (self.num_envs, 1)) + self.scene.env_origins
            #
            # self._tiled_camera.set_world_poses_from_view(eyes=eyes, targets=targets)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

         
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Store actions from policy in a class variable
        """
        self.last_action = self.robot_dof_targets[:, self.actuated_dof_indices]
        self.actions = actions.clone()
        
        
    def _apply_action(self) -> None:
        """
        The _apply_action(self) API is called decimation number of times for each RL step, prior to taking each physics step. 
        This provides more flexibility for environments where actions should be applied for each physics step.
        """
        self.robot_dof_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.robot_dof_lower_limits[self.actuated_dof_indices],
            self.robot_dof_upper_limits[self.actuated_dof_indices],
        )

        self.robot_dof_targets[:, self.actuated_dof_indices] = saturate(
            self.robot_dof_targets[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[self.actuated_dof_indices],
            self.robot_dof_upper_limits[self.actuated_dof_indices],
        )

        self.robot.set_joint_position_target(
            self.robot_dof_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        
        if self.cfg.obs_type == "gt":
            obs = self._get_gt()
        elif self.cfg.obs_type == "image":
            obs = self._get_images()
        elif self.cfg.obs_type == "image_prop":
            prop = self._get_proprioception()
            imgs = self._get_images()
            obs = torch.cat((prop, imgs), dim=-1)
        else:
            print("Unknown observations type!")

        return obs

    def _get_proprioception(self):
        prop = torch.cat((self.joint_pos, self.joint_vel, self.ee_pos), dim=-1)
        return prop

    def _get_gt(self):
        gt = torch.cat(
            (
                # robot
                self.joint_pos, 
                self.joint_vel,
                self.ee_pos,
                self.actions,
                # object
                self.object_pos,
                self.object_rot,
                # goal
                self.object_goal_distance.unsqueeze(1), # transform from (num_envs,) to (num_envs,1)
                # self.goal_pos,
                # self.goal_rot,
                # quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                
            ),
            dim=-1,
        )
        return gt

    def _get_images(self):
        if not self.init:
            # print(f'\n\nWITHIN _get_images(): {self.num_envs} // {self.cfg.eye} // {self.scene.env_origins}\n\n')
            eyes = torch.tensor(self.cfg.eye, dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) + self.scene.env_origins
            # eyes = torch.tensor([10, 22.5, 3.5], dtype=torch.float, device=self.device).repeat(
            #     (self.num_envs, 1)) + self.scene.env_origins

            targets = torch.tensor(self.cfg.target, dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) + self.scene.env_origins

            # print(f'eyes: {eyes} // {eyes.shape}')
            # print(f'targets: {targets} // {targets.shape}')
            # print(f'cam: {self._tiled_camera._view}\n')
            # self._tiled_camera.cfg.return_latest_camera_pose = True
            # self._tiled_camera.reset()
            self._tiled_camera.set_world_poses_from_view(eyes=eyes, targets=targets)
            # self.scene.sensors["tiled_camera"].set_world_poses_from_view(eyes=eyes, targets=targets)
            # self._tiled_camera.update(0, True)

            # print(f'cam: {self._tiled_camera._view}')
            self.init = True
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"

        # [num_envs, hw, hw, 3 (regardless of framestacking)]
        img_batch = self._tiled_camera.data.output[data_type].clone()
        # print(f'img_batch: {img_batch.shape}')
        # img_batch = self.scene.sensors["tiled_camera"].data.output[data_type].clone()
        batch_size = img_batch.size()[0]
        # flattened_images = img_batch.view(batch_size, -1)
        flattened_images = img_batch.reshape(img_batch.shape[0], -1)

        # sm = (img_batch - flattened_images.reshape(img_batch.shape)).abs().sum()
        # print(f'======\nRESHAPE: {sm}\n=======')
        # qqq

        # if self.cfg.write_image_to_file:
        #     name = self.count
        #     name = "lift"
        #     # img_dir = "/workspace/isaaclab/IsaacLabExtension/images/franka"
        #     # print(f'CURRENT FILE: {__file__}')
        #     img_dir = f"{__file__.replace('lift.py','')}"
        #     file_path = os.path.join(img_dir, f"{name}.png")
        #     save_images_to_file(img_batch, file_path)
            # self.count += 1

        # return img_batch
        return flattened_images

    def _get_rewards(self) -> torch.Tensor:
        # follow a curriculum
        if self.cfg.curriculum and self.common_step_counter > 10000:
            self.action_penalty_scale = -0.1
            self.joint_vel_penalty_scale = -0.1
        
        rewards, reaching_object, is_lifted, object_goal_tracking, object_goal_tracking_finegrained, action_rate_penalty, joint_vel_penalty = compute_rewards(
            self.reaching_object_scale,
            self.lift_object_scale, 
            self.object_goal_tracking_scale,
            self.object_goal_tracking_finegrained_scale,
            self.action_penalty_scale,
            self.joint_vel_penalty_scale,
            self.object_pos, self.joint_vel, self.actions, self.last_action, self.object_ee_distance, self.object_goal_distance, self.cfg.minimal_height)
        
        self.extras["log"] = {
        "reach_reward": (reaching_object).mean(),
        "lift_reward": (is_lifted).mean(),
        "object_goal_tracking": (object_goal_tracking).mean(),
        "object_goal_tracking_finegrained": (object_goal_tracking_finegrained).mean(),
        "action_rate": (action_rate_penalty).mean(),
        "joint_vel_penalty": (joint_vel_penalty).mean()
        }
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        
        # no termination at the moment
        out_of_reach = self.object_goal_distance >= self.cfg.fall_dist
        termination = out_of_reach

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset table (ISSUES! SO REMOVING FOR NOW)
        # table_state = self.table.data.default_root_state.clone()[env_ids]
        # self.table.write_root_state_to_sim(table_state, env_ids)


        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # reset robot
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # refresh intermediate values for _get_observations()
        self._compute_intermediate_values(env_ids)

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        # self.goal_markers.visualize(goal_pos, self.goal_rot)


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.ee_pos[env_ids] = self.ee_frame.data.target_pos_w[..., 0, :][env_ids] - self.scene.env_origins[env_ids]
                
        # object
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.object_rot[env_ids] = self.object.data.root_quat_w[env_ids]

        # relative distances
        self.object_ee_distance[env_ids] = torch.norm(self.object_pos[env_ids] - self.ee_pos[env_ids], dim=1)
        self.object_goal_distance[env_ids] = torch.norm(self.object_pos[env_ids] - self.goal_pos[env_ids], dim=1)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention

@torch.jit.script
def compute_rewards(
    reaching_object_scale: float,
    lift_object_scale: float, 
    object_goal_tracking_scale: float,
    object_goal_tracking_finegrained_scale: float,
    action_penalty_scale: float,
    joint_vel_penalty_scale: float,
    object_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    object_ee_distance: torch.Tensor,
    object_goal_distance: torch.Tensor,
    minimal_height: float,
):

    # reaching objects
    std = 0.1
    reaching_object = (1 - torch.tanh(object_ee_distance / std)) * reaching_object_scale
    reaching_object = -object_ee_distance * reaching_object_scale

    # reward for lifting object
    object_height = object_pos[:, 2]
    is_lifted = torch.where(object_height > minimal_height, 1.0, 0.0) * lift_object_scale

    # tracking
    std = 0.3
    object_goal_tracking = (object_height > minimal_height) * (1 - torch.tanh(object_goal_distance / std)) * object_goal_tracking_scale

    # fine tracking
    std = 0.05
    object_goal_tracking_finegrained = (object_height > minimal_height) * (1 - torch.tanh(object_goal_distance / std)) * object_goal_tracking_finegrained_scale

    # penalise rate of change of actions using L2 squared kernel
    action_rate_penalty = torch.sum(torch.square(action - last_action), dim=1) * action_penalty_scale

    # joint vel penalty l2
    joint_vel_penalty = torch.sum(torch.square(robot_joint_vel), dim=1) * joint_vel_penalty_scale
    
    rewards = (
        reaching_object
        + is_lifted * 0
        + object_goal_tracking * 0
        + object_goal_tracking_finegrained * 0
        + action_rate_penalty
        + joint_vel_penalty_scale
    )

    return rewards, reaching_object, is_lifted, object_goal_tracking, object_goal_tracking_finegrained, action_rate_penalty, joint_vel_penalty



