# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from multimodal_gym.utils.isaaclab.direct_rl_env import DirectRLElleEnv

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate


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

"""
    self.sim.physx.bounce_threshold_velocity = 0.2
    self.sim.physx.bounce_threshold_velocity = 0.01
    self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
    self.sim.physx.friction_correlation_distance = 0.00625
"""

@configclass
class ReachEnvCfg(DirectRLEnvCfg):
    # physics sim
    physics_dt = 0.01 # 100 Hz
    # number of physics step per control step
    decimation = 2
    # the number of physics simulation steps per rendering steps (default=1)
    render_interval = 2
    episode_length_s = 5.0  # 5 * 120 / 2 = 300 timesteps
    action_scale = 0.5  # [N]

    num_gt_observations = 30 # 7
    num_observations = 30
    num_actions = 9
    num_states = 0

    # lift stuff
    goal_height = 0.2

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1, replicate_physics=True)

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
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
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
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
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

    reaching_object_scale = 1
    lift_object_scale = 15
    object_goal_tracking_scale = 16.0
    object_goal_tracking_finegrained_scale = 5.0
    action_penalty_scale = -0.01
    joint_vel_penalty_scale = -0.01
    curriculum = False


class ReachEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    def __init__(self, cfg: ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.action_scale = self.cfg.action_scale
   

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # scale down the finger joint speed scales from 1 to 0.1
        self.left_finger_index =self.robot.find_joints("panda_finger_joint1")[0]
        self.right_finger_index =self.robot.find_joints("panda_finger_joint2")[0]

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.left_finger_index] = 0.1
        self.robot_dof_speed_scales[self.right_finger_index] = 0.1

        print(self.robot_dof_lower_limits)
        print(self.robot_dof_upper_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        self.joint_pos = torch.zeros((self.num_envs, 9), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, 9), device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ee_distance = torch.zeros((self.num_envs, ), device=self.device)

        self.reaching_object_scale = cfg.reaching_object_scale
        self.lift_object_scale = cfg.lift_object_scale
        self.object_goal_tracking_scale = cfg.object_goal_tracking_scale
        self.object_goal_tracking_finegrained_scale = cfg.object_goal_tracking_finegrained_scale
        self.action_penalty_scale = cfg.action_penalty_scale
        self.joint_vel_penalty_scale = cfg.joint_vel_penalty_scale


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        # FrameTransformer provides interface for reporting the transform of
        # one or more frames (target frames) wrt to another frame (source frame)
        self.ee_frame = FrameTransformer(self.cfg.ee_config)
        self.ee_frame.set_debug_vis(True)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["ee_frame"] = self.ee_frame

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

            
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

        self.robot.set_joint_position_target(self.robot_dof_targets)

        self.robot.set_joint_position_target(
            self.robot_dof_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

        # print(self.robot_dof_targets[0])
        
        # velocity_cmds = self.actions * 7.5 * self.robot_dof_speed_scales # * 0.5 # self.cfg.action_scale

        # targets = actions.clone() * self.robot_dof_upper_limits
        # check dt
        # check action scale
        # targets = self.robot_dof_targets + velocity_cmds * self.cfg.physics_dt

        # bool the fingers to open/close
        # targets[:, self.left_finger_index] = torch.where(self.actions[:, self.left_finger_index] >= 0.0, self.robot_dof_upper_limits[self.left_finger_index].item(),
        #                               self.robot_dof_lower_limits[self.left_finger_index].item())
        # targets[:, self.right_finger_index] = torch.where(self.actions[:, self.right_finger_index] >= 0.0, self.robot_dof_upper_limits[self.right_finger_index].item(),
        #                               self.robot_dof_lower_limits[self.right_finger_index].item())

        # self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        # print("actions: ", actions[0])
        # print("vel cmds: ", velocity_cmds[0])
        # print("targets: ", targets[0])
        # print("clamped targets: ", self.robot_dof_targets[0])
        # print("*******")

        # print("targets", self.robot_dof_targets[0][7:9])
        # self.last_action = self.actions.clone()

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos, 
                self.joint_vel,
                self.object_pos,
                self.actions
            ),
            dim=-1,
        )
        return obs

        
    def _get_rewards(self) -> torch.Tensor:

        # follow a curriculum
        if self.cfg.curriculum and self.common_step_counter > 10000:
            self.action_penalty_scale = -0.1
            self.joint_vel_penalty_scale = -0.1

        total_reward, reach_reward, action_rate_penalty, joint_vel_penalty = compute_rewards(
            self.reaching_object_scale,
            self.lift_object_scale, 
            self.object_goal_tracking_scale,
            self.object_goal_tracking_finegrained_scale,
            self.action_penalty_scale,
            self.joint_vel_penalty_scale,
            self.object_pos, self.joint_vel, self.actions, self.last_action, self.object_ee_distance, self.cfg.goal_height)
        
        self.extras["log"] = {
            "reach_reward": (reach_reward).mean(),
            "action_rate": (action_rate_penalty).mean(),
            "joint_vel_penalty": (joint_vel_penalty).mean()
        }


        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        
        # episode terminates if object reaches object pos
        reached = torch.where(self.object_ee_distance < 0, 1.0, 0.0)
        termination = reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return termination, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

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

        # reset object
        default_object_state = self.object.data.default_root_state[env_ids]
        default_object_state[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(default_object_state[:, :7], env_ids=env_ids)

        # refresh intermediate values for _get_observations()
        self._compute_intermediate_values(env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # robot data
        self.joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
        self.joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        
        ee_pos_world = self.ee_frame.data.target_pos_w[..., 0, :][env_ids]
        
        # panda_hand_index = self.robot.body_names.index('panda_hand')
        # self.ee_pos = self.robot.data.body_pos_w[:, panda_hand_index]  # ee position relative to robot base
        
        object_pos_world = self.object.data.root_pos_w[env_ids]
        self.object_ee_distance[env_ids] = torch.norm(object_pos_world - ee_pos_world, dim=1)
        # print(ee_w, object_ee_distance)

        # object data
        self.object_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        # self.object_rot = self.object.data.root_quat_w[env_ids]
        # self.object_velocities = self.object.data.root_vel_w[env_ids]
        # self.object_linvel = self.object.data.root_lin_vel_w[env_ids]
        # self.object_angvel = self.object.data.root_ang_vel_w[env_ids]

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


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
    goal_height: float,
):

    # reaching objects
    std = 0.1
    reaching_object = (1 - torch.tanh(object_ee_distance / std)) * reaching_object_scale
        
    # penalise rate of change of actions using L2 squared kernel
    action_rate_penalty = torch.sum(torch.square(action - last_action), dim=1) * action_penalty_scale
    # action_rate_penalty = torch.sum(torch.square(action), dim=-1)

    # joint vel penalty l2
    joint_vel_penalty = torch.sum(torch.square(robot_joint_vel), dim=1) * joint_vel_penalty_scale

    # timestep_penalty = -1
    
    rewards = (
        reaching_object 
        + action_rate_penalty 
        + joint_vel_penalty 
    )

    return rewards, reaching_object, action_rate_penalty, joint_vel_penalty



