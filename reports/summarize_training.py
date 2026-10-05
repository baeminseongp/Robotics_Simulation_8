"""Summarize iterations 900..999 per seed, then mean/sample std across seeds."""

import csv
from pathlib import Path
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for variant in ("ppo", "terrain", "friction", "curriculum", "history"):
        means = []
        for seed in range(3):
            runs = list((ROOT / "logs/rsl_rl" / f"ant_generalization_{variant}").glob(f"*_{variant}_seed{seed}"))
            if len(runs) != 1:
                raise ValueError(f"Expected one run: {variant} seed {seed}")
            events = EventAccumulator(str(runs[0]), size_guidance={"scalars": 0})
            events.Reload()
            items = [s for s in events.Scalars("Train/mean_reward") if 900 <= s.step <= 999]
            if sorted(s.step for s in items) != list(range(900, 1000)):
                raise ValueError(f"Incomplete/duplicate window: {runs[0]}")
            means.append(statistics.fmean(s.value for s in items))
        rows.append([variant, *means, statistics.fmean(means), statistics.stdev(means)])
    with (ROOT / "reports/training_ablation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "seed0_last100_mean", "seed1_last100_mean", "seed2_last100_mean", "mean", "sample_std"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
