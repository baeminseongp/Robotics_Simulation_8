# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-generalization variants of the classic Isaac Ant task."""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from ..ant_env_cfg import AntEnvCfg, EventCfg, MySceneCfg, ObservationsCfg
from . import mdp


@configclass
class CurriculumRandomUniformTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    function = mdp.curriculum_random_uniform_terrain


def terrain_generator_cfg(
    *, seed: int, curriculum: bool, difficulty_range: tuple[float, float] = (0.0, 1.0), unseen: bool = False
) -> TerrainGeneratorCfg:
    """Build the 20/30/30/20 terrain mixture from the experiment specification."""
    if unseen:
        noise = CurriculumRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.08, 0.12),
            noise_step=0.01,
            downsampled_scale=0.15,
            border_width=0.25,
        )
        boxes = terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.5, grid_width=0.30, grid_height_range=(0.08, 0.12), platform_width=2.0
        )
        sub_terrains = {"unseen_noise": noise, "unseen_blocks": boxes}
    else:
        sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
            "random_noise": CurriculumRandomUniformTerrainCfg(
                proportion=0.30,
                noise_range=(0.005, 0.10),
                noise_step=0.005,
                downsampled_scale=0.20,
                border_width=0.25,
            ),
            "random_blocks": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.30, grid_width=0.45, grid_height_range=(0.0, 0.10), platform_width=2.0
            ),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.20, slope_range=(0.0, 0.35), platform_width=2.0, border_width=0.25
            ),
        }
    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=curriculum,
        size=(8.0, 8.0),
        border_width=12.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.10,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=difficulty_range,
        use_cache=True,
        sub_terrains=sub_terrains,
    )


def terrain_importer_cfg(generator: TerrainGeneratorCfg, max_init_level: int | None = None) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=max_init_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


@configclass
class AntTerrainSceneCfg(MySceneCfg):
    terrain = terrain_importer_cfg(
        terrain_generator_cfg(seed=10_000, curriculum=False, difficulty_range=(0.0, 0.60))
    )


@configclass
class AntTerrainDREnvCfg(AntEnvCfg):
    """PPO with the specified terrain mixture and the original reward."""

    scene: AntTerrainSceneCfg = AntTerrainSceneCfg(num_envs=4096, env_spacing=8.0, clone_in_fabric=True)

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class RandomFrictionOnlyEventCfg(EventCfg):
    foot_friction = EventTerm(
        func=mdp.randomize_foot_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
            ),
            "static_friction_range": (0.4, 1.6),
            "dynamic_friction_range": (0.4, 1.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class FrictionEventCfg(RandomFrictionOnlyEventCfg):
    foot_friction = EventTerm(
        func=mdp.randomize_foot_material_per_env,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
            ),
            "static_friction_range": (0.4, 1.6),
            "dynamic_friction_range": (0.4, 1.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    def __post_init__(self):
        self.reset_base.params["pose_range"] = {
            "z": (-0.02, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }
        self.reset_base.params["velocity_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (-0.05, 0.05),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.10, 0.10),
        }
        self.reset_robot_joints.params["position_range"] = (-0.10, 0.10)
        self.reset_robot_joints.params["velocity_range"] = (-0.10, 0.10)


@configclass
class AntTerrainFrictionDREnvCfg(AntTerrainDREnvCfg):
    """Terrain DR plus episode-level contact-friction and reset randomization."""

    events: FrictionEventCfg = FrictionEventCfg()


@configclass
class AntCurriculumSceneCfg(MySceneCfg):
    terrain = terrain_importer_cfg(
        terrain_generator_cfg(seed=10_000, curriculum=True, difficulty_range=(0.0, 1.0)), max_init_level=1
    )


@configclass
class CurriculumEventCfg(FrictionEventCfg):
    foot_friction = EventTerm(
        func=mdp.curriculum_randomize_foot_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
            ),
            "stage_ranges": ((0.7, 1.3), (0.5, 1.5), (0.4, 1.6), (0.3, 1.7)),
            "num_buckets": 64,
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_forward)


@configclass
class AntCurriculumEnvCfg(AntTerrainFrictionDREnvCfg):
    """Full terrain range with performance-based terrain and friction curriculum."""

    scene: AntCurriculumSceneCfg = AntCurriculumSceneCfg(num_envs=4096, env_spacing=8.0, clone_in_fabric=True)
    events: CurriculumEventCfg = CurriculumEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class HistoryObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 4
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntHistoryEnvCfg(AntCurriculumEnvCfg):
    """Curriculum model with four policy-observation frames (240 values)."""

    observations: HistoryObservationsCfg = HistoryObservationsCfg()


@configclass
class FixedFrictionEventCfg(EventCfg):
    """Evaluation event with a fixed contact-friction coefficient."""

    foot_friction = EventTerm(
        func=mdp.randomize_foot_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
            ),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )


@configclass
class AntEvalSeenSceneCfg(MySceneCfg):
    terrain = terrain_importer_cfg(
        terrain_generator_cfg(seed=20_000, curriculum=False, difficulty_range=(0.0, 1.0))
    )


@configclass
class AntEvalSeenEnvCfg(AntTerrainDREnvCfg):
    scene: AntEvalSeenSceneCfg = AntEvalSeenSceneCfg(num_envs=100, env_spacing=8.0, clone_in_fabric=True)
    events: RandomFrictionOnlyEventCfg = RandomFrictionOnlyEventCfg()


@configclass
class AntEvalUnseenRoughSceneCfg(MySceneCfg):
    terrain = terrain_importer_cfg(
        terrain_generator_cfg(
            seed=30_000, curriculum=False, difficulty_range=(0.85, 1.0), unseen=True
        )
    )


@configclass
class AntEvalUnseenRoughEnvCfg(AntTerrainDREnvCfg):
    scene: AntEvalUnseenRoughSceneCfg = AntEvalUnseenRoughSceneCfg(
        num_envs=100, env_spacing=8.0, clone_in_fabric=True
    )
    events: FixedFrictionEventCfg = FixedFrictionEventCfg()


@configclass
class AntEvalUnseenSlipperyEnvCfg(AntEnvCfg):
    """Original flat task with fixed out-of-distribution contact friction 0.2."""

    events: FixedFrictionEventCfg = FixedFrictionEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.physics_material.friction_combine_mode = "multiply"
        self.events.foot_friction.params["static_friction_range"] = (0.2, 0.2)
        self.events.foot_friction.params["dynamic_friction_range"] = (0.2, 0.2)


@configclass
class AntEvalCombinedEnvCfg(AntEvalUnseenRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.foot_friction.params["static_friction_range"] = (0.2, 0.2)
        self.events.foot_friction.params["dynamic_friction_range"] = (0.2, 0.2)
