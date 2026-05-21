#!/usr/bin/env python3
"""Report per-video features and oracle deltas for base-vs-override rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from tools.gate_submission_hybrid import clear_rows_by_video, evaluate_video, split_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/raw/train/punches.csv"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--video-keys", help="Optional comma-separated video keys")
    parser.add_argument("--video-keys-from-predictions", action="store_true")
    parser.add_argument("--max-count-delta", type=int, default=48)
    parser.add_argument("--min-base-within-15", type=float, default=0.85)
    parser.add_argument("--min-override-within-15", type=float, default=0.70)
    parser.add_argument("--max-median-gap", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    gt_rows = read_csv_rows(args.ground_truth)
    base_rows = read_csv_rows(args.base)
    override_rows = read_csv_rows(args.override)
    keys = split_keys(args.video_keys)
    if args.video_keys_from_predictions:
        keys |= {row["video_key"] for row in base_rows}
        keys |= {row["video_key"] for row in override_rows}
    if keys:
        gt_rows = [row for row in gt_rows if row["video_key"] in keys]
        base_rows = [row for row in base_rows if row["video_key"] in keys]
        override_rows = [row for row in override_rows if row["video_key"] in keys]

    base_score = score_predictions(gt_rows, base_rows)["by_video"]
    override_score = score_predictions(gt_rows, override_rows)["by_video"]
    base_by_video = clear_rows_by_video(base_rows)
    override_by_video = clear_rows_by_video(override_rows)
    gate_args = SimpleNamespace(
        max_count_delta=args.max_count_delta,
        min_base_within_15=args.min_base_within_15,
        min_override_within_15=args.min_override_within_15,
        max_median_gap=args.max_median_gap,
    )
    print(
        "video_key,data_root,fight_index,round_number,base_score,override_score,delta,"
        "base_count,override_count,count_delta,base_le15,override_le15,"
        "base_med_gap,override_med_gap,gate_pass,oracle_pick"
    )
    for key in sorted(set(base_by_video) | set(override_by_video)):
        if key not in base_score or key not in override_score:
            continue
        video = video_by_key[key]
        gate = evaluate_video(key, base_by_video.get(key, []), override_by_video.get(key, []), gate_args)
        base_value = base_score[key]["final_score"]
        override_value = override_score[key]["final_score"]
        print(
            f"{key},{video['data_root']},{video['fight_index']},{video['round_number']},"
            f"{base_value:.6f},{override_value:.6f},{override_value - base_value:+.6f},"
            f"{gate.base_count},{gate.override_count},{gate.count_delta},"
            f"{gate.base_within_15:.4f},{gate.override_within_15:.4f},"
            f"{gate.base_median_gap:.1f},{gate.override_median_gap:.1f},"
            f"{int(gate.pass_gate)},{'override' if override_value > base_value else 'base'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
