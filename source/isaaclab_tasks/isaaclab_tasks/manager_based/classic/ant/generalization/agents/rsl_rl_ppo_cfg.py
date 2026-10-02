# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntGeneralizationPPORunnerCfg(AntPPORunnerCfg):
    """The original Ant PPO hyperparameters with isolated generalization logs."""

    experiment_name = "ant_generalization"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

