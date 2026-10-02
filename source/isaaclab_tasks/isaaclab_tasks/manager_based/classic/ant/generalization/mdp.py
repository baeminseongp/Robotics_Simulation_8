# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP helpers for Ant terrain generalization."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.height_field import hf_terrains

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import HfRandomUniformTerrainCfg


def curriculum_random_uniform_terrain(difficulty: float, cfg: HfRandomUniformTerrainCfg):
    """Generate non-negative noise whose amplitude grows with terrain difficulty."""
    local_cfg = copy.deepcopy(cfg)
    min_amplitude, max_amplitude = cfg.noise_range
    amplitude = min_amplitude + difficulty * (max_amplitude - min_amplitude)
    local_cfg.noise_range = (0.0, max(amplitude, cfg.vertical_scale))
    return hf_terrains.random_uniform_terrain(difficulty, local_cfg)


def terrain_levels_forward(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    promote_distance: float = 3.0,
    demote_distance: float = 0.75,
) -> torch.Tensor:
    """Promote Ants that travel forward and demote Ants that make little progress."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    displacement = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    forward_distance = displacement[:, 0]
    move_up = forward_distance > promote_distance
    move_down = forward_distance < demote_distance
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


class randomize_foot_material_per_env(randomize_rigid_body_material):
    """Assign one contact-friction bucket to all selected feet in each environment."""

    def _apply_bucket_ids(self, env_ids: torch.Tensor, bucket_ids: torch.Tensor):
        total_num_shapes = self.asset.root_physx_view.max_shapes
        material_samples = self.material_buckets[bucket_ids[:, None].expand(-1, total_num_shapes)]
        materials = self.asset.root_physx_view.get_material_properties()
        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids] = material_samples
        self.asset.root_physx_view.set_material_properties(materials, env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        del static_friction_range, dynamic_friction_range, restitution_range, asset_cfg, make_consistent
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids),), device="cpu")
        self._apply_bucket_ids(env_ids, bucket_ids)


class curriculum_randomize_foot_material(randomize_foot_material_per_env):
    """Assign per-environment foot friction using ranges determined by terrain level."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        stage_ranges = cfg.params["stage_ranges"]
        low = min(item[0] for item in stage_ranges)
        high = max(item[1] for item in stage_ranges)
        values = torch.linspace(low, high, int(cfg.params["num_buckets"]), device="cpu")
        self.material_buckets[:, 0] = values
        self.material_buckets[:, 1] = values
        self.material_buckets[:, 2] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        stage_ranges: tuple[tuple[float, float], ...],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
    ):
        del asset_cfg
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        terrain: TerrainImporter = env.scene.terrain
        levels = terrain.terrain_levels[env_ids.to(terrain.terrain_levels.device)].cpu()
        stage_ids = torch.clamp(
            levels * len(stage_ranges) // max(terrain.max_terrain_level, 1), max=len(stage_ranges) - 1
        )
        bucket_ids_per_env = []
        for stage_id in stage_ids.tolist():
            stage_low, stage_high = stage_ranges[stage_id]
            low_id = int(torch.searchsorted(self.material_buckets[:, 0], torch.tensor(stage_low)).item())
            high_id = int(torch.searchsorted(self.material_buckets[:, 0], torch.tensor(stage_high), right=True).item())
            low_id = max(0, min(low_id, num_buckets - 1))
            high_id = max(low_id + 1, min(high_id, num_buckets))
            bucket_ids_per_env.append((low_id, high_id))

        bucket_ids = torch.empty(len(env_ids), dtype=torch.long, device="cpu")
        for row, (low_id, high_id) in enumerate(bucket_ids_per_env):
            bucket_ids[row] = torch.randint(low_id, high_id, (), device="cpu")
        self._apply_bucket_ids(env_ids, bucket_ids)
