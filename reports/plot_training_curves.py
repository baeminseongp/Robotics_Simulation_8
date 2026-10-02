"""Create the training-return curve from the submitted TensorBoard logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs" / "rsl_rl"
VARIANTS = ("ppo", "terrain", "friction", "curriculum", "history")


def load_curve(run: Path) -> tuple[np.ndarray, np.ndarray]:
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    scalars = events.Scalars("Train/mean_reward")
    return np.asarray([item.step for item in scalars]), np.asarray([item.value for item in scalars])


def main() -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for variant in VARIANTS:
        runs = sorted((LOG_ROOT / f"ant_generalization_{variant}").glob(f"*_{variant}_seed*"))
        curves = [load_curve(run) for run in runs]
        common_steps = curves[0][0]
        values = np.stack([np.interp(common_steps, steps, rewards) for steps, rewards in curves])
        for seed_curve in values:
            axis.plot(common_steps, seed_curve, alpha=0.16, linewidth=0.8)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axis.plot(common_steps, mean, linewidth=2.0, label=variant)
        axis.fill_between(common_steps, mean - std, mean + std, alpha=0.12)

    axis.set_title("Isaac-Ant training return (3 seeds)")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Train/mean_reward")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    figure.savefig(ROOT / "reports" / "training_return_curve.png", dpi=180)


if __name__ == "__main__":
    main()
