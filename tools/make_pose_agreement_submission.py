#!/usr/bin/env python3
"""Generate a submission from normalized agreement between two pose caches."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, estimate_attrs, fit_attr_priors, select_candidates
from rascar_boxing.validation import validate_submission
from tools.evaluate_pose_model_agreement import (
    agreement_arrays,
    combine_candidates,
    normalized_candidates,
)
from tools.evaluate_pose_selection_variants import estimate_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--primary-tracks-dir", type=Path, required=True)
    parser.add_argument("--secondary-tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/pose_agreement.csv"))
    parser.add_argument("--max-candidates-per-source-video", type=int, default=4000)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--agreement-alpha", type=float, default=0.2)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--secondary-weight", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=1.4)
    parser.add_argument("--nms-frames", type=int, default=10)
    parser.add_argument("--cross-nms-frames", type=int, default=2)
    parser.add_argument("--count-mode", default="root_count")
    parser.add_argument("--count-multiplier", type=float, default=0.88)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_videos = read_csv_rows(args.data_root / "train/videos.csv")
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    punches = [row for row in read_csv_rows(args.data_root / "train/punches.csv") if row["clear"] == "true"]
    sample_rows = read_csv_rows(args.data_root / "sample_submission.csv")

    train_counts = Counter(row["video_key"] for row in punches)
    attr_priors = fit_attr_priors(punches)
    selected_by_video = {}

    for video in sorted(test_videos, key=lambda row: row["video_key"]):
        key = video["video_key"]
        primary = normalized_candidates(
            args.primary_tracks_dir / f"{key}.jsonl",
            "primary",
            args.pool_min_score,
            args.pool_nms_frames,
            args.max_candidates_per_source_video,
        )
        secondary = normalized_candidates(
            args.secondary_tracks_dir / f"{key}.jsonl",
            "secondary",
            args.pool_min_score,
            args.pool_nms_frames,
            args.max_candidates_per_source_video,
        )
        agreement = agreement_arrays(primary, secondary, args.window)
        combined = combine_candidates(
            primary,
            secondary,
            agreement,
            args.agreement_alpha,
            args.primary_weight,
            args.secondary_weight,
        )
        count = estimate_count(
            video,
            train_videos,
            train_counts,
            Counter(),
            args.count_mode,
            args.count_multiplier,
        )
        selected_by_video[key] = select_candidates(
            combined,
            PoseHeuristicConfig(
                min_score=args.threshold,
                nms_frames=args.nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=args.cross_nms_frames,
            ),
            count,
        )

    rows = fill_sample_rows(sample_rows, test_videos, selected_by_video, attr_priors)
    write_csv_rows(args.output, rows, SUBMISSION_COLUMNS)
    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1

    by_video: dict[str, int] = {}
    for row in rows:
        if row["clear"] == "true":
            by_video[row["video_key"]] = by_video.get(row["video_key"], 0) + 1
    print(f"Wrote {args.output}")
    print(
        "config="
        f"window={args.window},agreement_alpha={args.agreement_alpha},"
        f"primary_weight={args.primary_weight},secondary_weight={args.secondary_weight},"
        f"threshold={args.threshold},nms={args.nms_frames},cross={args.cross_nms_frames},"
        f"count_mode={args.count_mode},count_multiplier={args.count_multiplier}"
    )
    print("selected=" + ",".join(f"{key}:{by_video.get(key, 0)}" for key in sorted(by_video)))
    return 0


def fill_sample_rows(
    sample_rows: list[dict[str, str]],
    videos: list[dict[str, str]],
    selected_by_video: dict[str, list],
    attr_priors: dict[str, object],
) -> list[dict[str, str]]:
    videos_by_key = {row["video_key"]: row for row in videos}
    used_by_video: Counter[str] = Counter()
    output = []
    for sample_row in sample_rows:
        row = {column: sample_row[column] for column in SUBMISSION_COLUMNS}
        video_key = row["video_key"]
        index = used_by_video[video_key]
        selected = selected_by_video.get(video_key, [])
        if index < len(selected):
            candidate = selected[index]
            video = videos_by_key[video_key]
            attrs = estimate_attrs(candidate, attr_priors)
            row["frame"] = str(max(0, min(int(video["frame_count"]) - 1, candidate.frame)))
            row["fighter"] = candidate.fighter
            row["punch_type"] = attrs["punch_type"]
            row["hand"] = candidate.hand
            row["target"] = candidate.target
            row["effectiveness"] = attrs["effectiveness"]
            row["clear"] = "true"
        else:
            row["clear"] = "false"
        used_by_video[video_key] += 1
        output.append(row)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
