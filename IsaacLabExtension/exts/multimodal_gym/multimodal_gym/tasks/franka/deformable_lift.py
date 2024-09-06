# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import DeformableObjectCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.config.franka import joint_pos_env_cfg
import math
import numpy as np
import torch
from collections.abc import Sequence
import os
from multimodal_gym.utils.isaaclab.direct_rl_env import DirectRLElleEnv
import gymnasium as gym

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, DeformableObjectCfg, DeformableObject
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
##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

from multimodal_gym.tasks.franka.lift import LiftEnvCfg, LiftEnv


@configclass
class DeformableLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.replicate_physics = False

        FRANKA_PANDA_CFG.actuators["panda_hand"].effort_limit = 2.0

        
    
        # self.events.reset_object_position = EventTerm(
        #     func=mdp.reset_nodal_state_uniform,
        #     mode="reset",
        #     params={
        #         "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
        #         "velocity_range": {},
        #         "asset_cfg": SceneEntityCfg("object"),
        #     },
        #     )


class DeformableLiftEnv(LiftEnv):
    cfg: DeformableLiftEnvCfg

    def __init__(self, cfg: DeformableLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        def_object_cfg = DeformableObjectCfg(
            prim_path="/World/envs/env_.*/DefObject",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.5], rot=[1, 0, 0, 0]),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.06, 0.06, 0.06),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                    self_collision_filter_distance=0.005,
                    settling_threshold=0.1,
                    sleep_damping=1.0,
                    sleep_threshold=0.05,
                    solver_position_iteration_count=20,
                    vertex_velocity_damping=0.5,
                    simulation_hexahedral_resolution=4,
                    rest_offset=0.0001,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    dynamic_friction=0.95,
                    youngs_modulus=500000,
                ),
            ),
        )
        self.object = DeformableObject(def_object_cfg)

        # FrameTransformer provides interface for reporting the transform of
        # one or more frames (target frames) wrt to another frame (source frame)
        self.ee_frame = FrameTransformer(self.cfg.ee_config)
        self.ee_frame.set_debug_vis(True)
        # self.goal_frame = FrameTransformer(self.cfg.goal_object_cfg)
        # self.goal_frame.set_debug_vis(True)
        # self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

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

        if self.cfg.obs_type == "image" or self.cfg.obs_type == "image_prop":
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        nodal_state = self.object.data.default_nodal_state_w.clone()[env_ids]
        self.object.write_nodal_state_to_sim(nodal_state, env_ids)

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