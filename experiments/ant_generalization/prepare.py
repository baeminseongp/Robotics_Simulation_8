"""Prepare an isolated-source Ant baseline run without changing installed packages.

Default: print the command. --check: inspect dependencies. --execute: run PPO.
Use the Isaac Sim Python interpreter to run this script.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
PACKAGES = ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")
TASKS = {
    "ppo": "Isaac-Ant-v0",
    "terrain": "Isaac-Ant-Generalization-Terrain-v0",
    "friction": "Isaac-Ant-Generalization-Friction-v0",
    "curriculum": "Isaac-Ant-Generalization-Curriculum-v0",
    "history": "Isaac-Ant-Generalization-History-v0",
}


def source_environment():
    env = os.environ.copy()
    paths = [str(ROOT / "source" / package) for package in PACKAGES]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def inspect_runtime():
    report = {"python": sys.version, "executable": sys.executable, "packages": {}, "sources": {}, "errors": []}
    for name in ("isaacsim", "torch", "rsl-rl-lib", "gymnasium", "hydra-core", "tensorboard"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing package: {name}")
    for name in PACKAGES:
        spec = importlib.util.find_spec(name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        report["sources"][name] = str(origin)
        if origin is None or not origin.is_relative_to(ROOT / "source" / name):
            report["errors"].append(f"Unexpected source for {name}: {origin}")
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            report["gpu"] = torch.cuda.get_device_name(0)
        else:
            report["errors"].append("CUDA is unavailable")
    except ImportError as exc:
        report["errors"].append(str(exc))
    report["note"] = "Metadata/source/CUDA check only; simulator and PPO compatibility require a smoke run."
    print(json.dumps(report, indent=2))
    return bool(report["errors"])


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--inspect-runtime", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="32 environments, 2 iterations; separate smoke logs")
    parser.add_argument("--variant", choices=tuple(TASKS), default="ppo")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--num-envs", type=positive_int)
    args = parser.parse_args()
    if args.inspect_runtime:
        return inspect_runtime()
    env = source_environment()
    check_command = [sys.executable, str(Path(__file__).resolve()), "--inspect-runtime"]
    if args.check:
        return subprocess.call(check_command, cwd=ROOT, env=env)
    protocol = json.loads(PROTOCOL.read_text())
    train = protocol["training"]
    count = args.num_envs or (32 if args.smoke else train["num_envs"])
    # Keep the transition budget comparable when reducing environment count.
    budget = train["transitions_per_seed"]
    batch_size = count * train["steps_per_env"]
    if not args.smoke and budget % batch_size:
        parser.error("num-envs must divide the fixed transition budget exactly")
    iterations = 2 if args.smoke else budget // batch_size
    experiment = "ant_generalization_smoke" if args.smoke else f"ant_generalization_{args.variant}"
    command = [
        sys.executable, str(ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"),
        "--task", TASKS[args.variant], "--headless", "--device", "cuda:0",
        "--num_envs", str(count), "--max_iterations", str(iterations),
        "--seed", str(args.seed), "--experiment_name", experiment,
        "--run_name", f"{args.variant}_seed{args.seed}", "--logger", "tensorboard",
    ]
    print(f"Working directory: {ROOT}", flush=True)
    print(f"PYTHONPATH={shlex.quote(env['PYTHONPATH'])} {shlex.join(command)}", flush=True)
    print(f"Transitions: {count * iterations * train['steps_per_env']:,}", flush=True)
    if not args.execute:
        print("Dry run only. Add --execute to launch.")
        return 0
    result = subprocess.call(check_command, cwd=ROOT, env=env)
    if result:
        return result
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
