#!/usr/bin/env python3
"""Evaluate temporal-prior variants on a train-video split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.temporal_prior import TemporalPriorConfig, fit_temporal_prior, predict_for_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--val-keys",
        type=Path,
        default=Path("data/processed/splits/fight_seed42/val_video_keys.txt"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    val_keys = set(args.val_keys.read_text(encoding="utf-8").split())
    train_keys = {row["video_key"] for row in videos} - val_keys
    val_videos = [row for row in videos if row["video_key"] in val_keys]
    val_gt = [row for row in punches if row["video_key"] in val_keys]

    results = []
    for group_cols in [("dataset_type",), ("dataset_type", "round_number"), ("data_root", "round_number")]:
        for count_mode in ["count", "rate"]:
            for spacing in ["quantile", "uniform"]:
                for fighter_mode in ["blue", "alternate"]:
                    for multiplier in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]:
                        config = TemporalPriorConfig(
                            group_cols=group_cols,
                            count_mode=count_mode,
                            spacing=spacing,
                            fighter_mode=fighter_mode,
                            count_multiplier=multiplier,
                        )
                        prior = fit_temporal_prior(videos, punches, train_keys, config)
                        pred = predict_for_videos(prior, val_videos)
                        score = score_predictions(val_gt, pred)["macro_score"]
                        results.append((score, len(pred), config))

    print("score,n_pred,group_cols,count_mode,spacing,fighter_mode,count_multiplier")
    for score, n_pred, config in sorted(results, key=lambda item: item[0], reverse=True)[: args.top_k]:
        print(
            f"{score:.6f},{n_pred},{'|'.join(config.group_cols)},"
            f"{config.count_mode},{config.spacing},{config.fighter_mode},{config.count_multiplier}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

