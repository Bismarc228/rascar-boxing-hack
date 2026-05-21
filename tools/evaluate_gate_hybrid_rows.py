#!/usr/bin/env python3
"""Sweep per-video gate thresholds for two validation prediction row files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from tools.evaluate_pose_selection_variants import score_summary, video_wins
from tools.gate_submission_hybrid import clear_rows_by_video, evaluate_video, split_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=Path("data/raw/train/punches.csv"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--video-keys", help="Optional comma-separated video keys to score")
    parser.add_argument("--video-keys-from-predictions", action="store_true")
    parser.add_argument("--max-count-deltas", default="0,4,8,12,16,20,24,32,48")
    parser.add_argument("--min-base-within-15", default="0.50,0.60,0.70,0.80,0.85,0.90")
    parser.add_argument("--min-override-within-15", default="0.50,0.60,0.70,0.80,0.85,0.90")
    parser.add_argument("--max-median-gaps", default="1,2,4,8,15,30")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_rows = read_csv_rows(args.base)
    override_rows = read_csv_rows(args.override)
    gt_rows = read_csv_rows(args.ground_truth)
    video_keys = split_keys(args.video_keys)
    if args.video_keys_from_predictions:
        video_keys |= {row["video_key"] for row in base_rows}
        video_keys |= {row["video_key"] for row in override_rows}
    if video_keys:
        base_rows = [row for row in base_rows if row["video_key"] in video_keys]
        override_rows = [row for row in override_rows if row["video_key"] in video_keys]
        gt_rows = [row for row in gt_rows if row["video_key"] in video_keys]

    base_score = score_predictions(gt_rows, base_rows)
    override_score = score_predictions(gt_rows, override_rows)
    base_by_video = clear_rows_by_video(base_rows)
    override_by_video = clear_rows_by_video(override_rows)
    keys = sorted((set(base_by_video) | set(override_by_video)) & {row["video_key"] for row in gt_rows})
    results = []
    for max_count_delta in parse_ints(args.max_count_deltas):
        for min_base in parse_floats(args.min_base_within_15):
            for min_override in parse_floats(args.min_override_within_15):
                for max_median_gap in parse_floats(args.max_median_gaps):
                    gate_args = SimpleNamespace(
                        max_count_delta=max_count_delta,
                        min_base_within_15=min_base,
                        min_override_within_15=min_override,
                        max_median_gap=max_median_gap,
                    )
                    gates = [
                        evaluate_video(
                            key,
                            base_by_video.get(key, []),
                            override_by_video.get(key, []),
                            gate_args,
                        )
                        for key in keys
                    ]
                    replace_keys = {gate.video_key for gate in gates if gate.pass_gate}
                    rows = hybrid_rows(base_rows, override_rows, replace_keys)
                    score = score_predictions(gt_rows, rows)
                    summary = score_summary(score)
                    results.append(
                        (
                            score["macro_score"],
                            summary["time"],
                            summary["fp_penalty"],
                            video_wins(score, base_score),
                            video_wins(score, override_score),
                            len(rows),
                            max_count_delta,
                            min_base,
                            min_override,
                            max_median_gap,
                            ",".join(sorted(replace_keys)),
                        )
                    )

    print(
        f"base_score={base_score['macro_score']:.6f} "
        f"override_score={override_score['macro_score']:.6f} videos={len(keys)}"
    )
    print(
        "score,time,fp,wins_vs_base,wins_vs_override,n_rows,max_count_delta,"
        "min_base15,min_override15,max_median_gap,replace_keys"
    )
    for result in sorted(results, reverse=True)[: args.top_k]:
        print(
            f"{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},{result[3]},"
            f"{result[4]},{result[5]},{result[6]},{result[7]},{result[8]},"
            f"{result[9]},{result[10]}"
        )
    return 0


def hybrid_rows(
    base_rows: list[dict[str, str]],
    override_rows: list[dict[str, str]],
    replace_keys: set[str],
) -> list[dict[str, str]]:
    rows = [dict(row) for row in base_rows if row["video_key"] not in replace_keys]
    rows.extend(dict(row) for row in override_rows if row["video_key"] in replace_keys)
    rows.sort(key=lambda row: (row["video_key"], int(row["frame"]), row["fighter"], row["hand"]))
    for index, row in enumerate(rows, start=1):
        row["id"] = str(index)
    return rows


def parse_floats(values: str) -> list[float]:
    return [float(value) for value in values.split(",") if value]


def parse_ints(values: str) -> list[int]:
    return [int(value) for value in values.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
