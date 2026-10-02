# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Registered Ant terrain-generalization training and evaluation environments."""

import gymnasium as gym

from . import agents


_TASKS = {
    "Isaac-Ant-Generalization-Terrain-v0": "AntTerrainDREnvCfg",
    "Isaac-Ant-Generalization-Friction-v0": "AntTerrainFrictionDREnvCfg",
    "Isaac-Ant-Generalization-Curriculum-v0": "AntCurriculumEnvCfg",
    "Isaac-Ant-Generalization-History-v0": "AntHistoryEnvCfg",
    "Isaac-Ant-Generalization-Eval-Seen-v0": "AntEvalSeenEnvCfg",
    "Isaac-Ant-Generalization-Eval-Unseen-Rough-v0": "AntEvalUnseenRoughEnvCfg",
    "Isaac-Ant-Generalization-Eval-Unseen-Slippery-v0": "AntEvalUnseenSlipperyEnvCfg",
    "Isaac-Ant-Generalization-Eval-Combined-v0": "AntEvalCombinedEnvCfg",
}

for task_id, cfg_name in _TASKS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_generalization_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGeneralizationPPORunnerCfg",
        },
    )

