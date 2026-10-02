"""Run or print the core Ant generalization ablation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


VARIANTS = ("ppo", "terrain", "friction", "curriculum", "history")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args()

    prepare = Path(__file__).with_name("prepare.py")
    for variant in args.variants:
        for seed in args.seeds:
            command = [sys.executable, str(prepare), "--variant", variant, "--seed", str(seed)]
            if args.smoke:
                command.append("--smoke")
            if args.num_envs:
                command.extend(("--num-envs", str(args.num_envs)))
            if args.execute:
                command.append("--execute")
            print(" ".join(command), flush=True)
            result = subprocess.call(command)
            if result:
                return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
