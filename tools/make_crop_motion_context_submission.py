#!/usr/bin/env python3
"""Generate a pose-context submission with raw crop-motion rescoring."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    estimate_count,
    fill_sample_rows,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from rascar_boxing.validation import validate_submission
from tools.evaluate_crop_motion_context import compute_all_motion, rescore_with_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--nms-frames", type=int, default=10)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--nms-group-mode", default="fighter_hand")
    parser.add_argument("--count-mode", default="root_rate")
    parser.add_argument("--count-multiplier", type=float, default=0.88)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--motion-alpha", type=float, default=-0.04)
    parser.add_argument("--motion-beta", type=float, default=0.08)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_videos = read_csv_rows(args.data_root / "train/videos.csv")
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    train_punches = read_csv_rows(args.data_root / "train/punches.csv")
    sample_rows = read_csv_rows(args.data_root / "sample_submission.csv")
    attr_priors = fit_attr_priors(train_punches)
    video_by_key = {row["video_key"]: row for row in test_videos}
    ready_keys = [
        row["video_key"]
        for row in test_videos
        if (args.tracks_dir / f"{row['video_key']}.jsonl").exists()
    ]
    if len(ready_keys) != len(test_videos):
        missing = sorted(set(video_by_key) - set(ready_keys))
        raise FileNotFoundError(f"Missing test tracks: {missing}")

    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode=args.nms_group_mode,
        cross_nms_frames=args.cross_nms_frames,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )

    candidates_by_key: dict[str, list[PunchCandidate]] = {}
    counts_by_key: dict[str, int | None] = {}
    for key in ready_keys:
        raw_candidates = score_pose_tracks(
            args.tracks_dir / f"{key}.jsonl",
            PoseHeuristicConfig(min_score=0.0),
        )
        candidates_by_key[key] = apply_temporal_context(raw_candidates, config)
        counts_by_key[key] = estimate_count(config, video_by_key[key], train_videos, train_punches, sample_rows)

    motion_by_key = compute_all_motion(
        args.data_root,
        args.tracks_dir,
        ready_keys,
        video_by_key,
        candidates_by_key,
        args.resize_width,
        args.quiet,
        args.jobs,
    )
    selected_by_video = {}
    for key in ready_keys:
        rescored = [
            rescore_with_motion(
                candidate,
                motion_by_key[key].get(candidate.frame, (0.0, 0.0, 0.0)),
                args.motion_alpha,
                args.motion_beta,
            )
            for candidate in candidates_by_key[key]
        ]
        selected_by_video[key] = select_candidates(rescored, config, counts_by_key[key])

    rows = fill_sample_rows(sample_rows, test_videos, selected_by_video, attr_priors)
    write_csv_rows(args.output, rows, SUBMISSION_COLUMNS)
    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Wrote {args.output}")
    print(
        "selected="
        + ",".join(
            f"{key}:{len(selected_by_video.get(key, []))}" for key in sorted(selected_by_video)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
