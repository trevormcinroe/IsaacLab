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
##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

from multimodal_gym.tasks.franka.lift import LiftEnvCfg, LiftEnv


@configclass
class OccludedLiftEnvCfg(LiftEnvCfg):
    
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.35], rot=[1, 0, 0, 0]),
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
    box_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.3], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
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
    clutter_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Clutter",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.5], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/003_cracker_box.usd",
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



class OccludedLiftEnv(LiftEnv):
    cfg: OccludedLiftEnvCfg

    def __init__(self, cfg: OccludedLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        self.box = RigidObject(self.cfg.box_cfg)
        self.clutter = RigidObject(self.cfg.clutter_cfg)
        self.scene.rigid_objects["box"] = self.box
        self.scene.rigid_objects["clutter"] = self.clutter
        print("added box and clutter")

    def reset_idx(self, env_ids: Sequence[int] | None):
        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._reset_idx()

        box_default_state = self.box.data.default_root_state.clone()[env_ids]
        self.box.write_root_state_to_sim(box_default_state, env_ids)

        clutter_default_state = self.clutter.data.default_root_state.clone()[env_ids]
        self.clutter.write_root_state_to_sim(clutter_default_state, env_ids)
