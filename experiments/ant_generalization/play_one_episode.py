"""Play one episode in each environment and report the return mean and std.

This is the assignment evaluator. With the default ``--num_envs 100`` it
collects exactly one completed episode from each of 100 parallel environments.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import statistics

from isaaclab.app import AppLauncher


SUITES = {
    "flat": "Isaac-Ant-v0",
    "seen": "Isaac-Ant-Generalization-Eval-Seen-v0",
    "unseen_rough": "Isaac-Ant-Generalization-Eval-Unseen-Rough-v0",
    "unseen_slippery": "Isaac-Ant-Generalization-Eval-Unseen-Slippery-v0",
    "combined": "Isaac-Ant-Generalization-Eval-Combined-v0",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--suite", choices=tuple(SUITES), default="combined")
parser.add_argument("--history", action="store_true", help="Use four-frame policy observations.")
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--eval_seed", type=int, default=30_000)
parser.add_argument("--output", type=Path, help="Optional JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


def main() -> int:
    if args.num_envs != 100:
        print(f"[warning] The assignment requires --num_envs 100; received {args.num_envs}.")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    task = SUITES[args.suite]
    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.eval_seed
    if args.history:
        env_cfg.observations.policy.history_length = 4
        env_cfg.observations.policy.flatten_history_dim = True

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device

    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)

    observations = env.get_observations()
    running_returns = torch.zeros(env.num_envs, device=env.device)
    completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    episode_returns = torch.empty(env.num_envs, device=env.device)
    step_count = 0

    while not bool(completed.all()) and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, _ = env.step(actions)
        dones = dones.bool()
        running_returns += rewards
        newly_completed = dones & ~completed
        episode_returns[newly_completed] = running_returns[newly_completed]
        completed |= newly_completed
        policy.reset(dones)
        step_count += 1
        if step_count % 100 == 0:
            print(f"[evaluation] step={step_count}, completed={int(completed.sum().item())}/{env.num_envs}", flush=True)

    env.close()
    if not bool(completed.all()):
        raise RuntimeError("Simulation stopped before all environments completed one episode.")

    values = episode_returns.detach().cpu().tolist()
    summary = {
        "source": "play_one_episode.py",
        "suite": args.suite,
        "task": task,
        "checkpoint": str(checkpoint),
        "eval_seed": args.eval_seed,
        "num_envs": args.num_envs,
        "episodes": len(values),
        "episode_return_mean": statistics.fmean(values),
        "episode_return_std": statistics.stdev(values),
        "episode_returns": values,
    }

    print("\n=== One episode per environment ===")
    print(f"suite: {summary['suite']}")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"episodes: {summary['episodes']}")
    print(f"episode return mean: {summary['episode_return_mean']:.6f}")
    print(f"episode return std:  {summary['episode_return_std']:.6f}")

    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
