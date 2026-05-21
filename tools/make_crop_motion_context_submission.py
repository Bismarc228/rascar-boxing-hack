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
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    estimate_attrs,
    estimate_count,
    fill_sample_rows,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from rascar_boxing.validation import validate_submission
from tools.evaluate_crop_motion_context import compute_all_motion, rescore_with_motion
from tools.evaluate_pose_selection_variants import score_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--video-keys", help="Optional comma-separated validation/test keys")
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
    parser.add_argument("--score", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_videos = read_csv_rows(args.data_root / "train/videos.csv")
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    all_punches = read_csv_rows(args.data_root / "train/punches.csv")
    sample_rows = read_csv_rows(args.data_root / "sample_submission.csv")
    requested_keys = {value for value in (args.video_keys or "").split(",") if value}
    if args.split == "test":
        videos = [row for row in test_videos if not requested_keys or row["video_key"] in requested_keys]
        train_rows = train_videos
        punch_rows = all_punches
        gt_rows = []
        ready_keys = [
            row["video_key"]
            for row in videos
            if (args.tracks_dir / f"{row['video_key']}.jsonl").exists()
        ]
        if len(ready_keys) != len(videos):
            missing = sorted({row["video_key"] for row in videos} - set(ready_keys))
            raise FileNotFoundError(f"Missing test tracks: {missing}")
    else:
        video_by_key_all = {row["video_key"]: row for row in train_videos}
        ready_keys = complete_track_keys(args.tracks_dir, video_by_key_all)
        if requested_keys:
            ready_keys = [key for key in ready_keys if key in requested_keys]
        if not ready_keys:
            print("No complete validation tracks found", file=sys.stderr)
            return 2
        ready_set = set(ready_keys)
        videos = [row for row in train_videos if row["video_key"] in ready_set]
        train_rows = [row for row in train_videos if row["video_key"] not in ready_set]
        punch_rows = [
            row for row in all_punches if row["clear"] == "true" and row["video_key"] not in ready_set
        ]
        gt_rows = [
            row for row in all_punches if row["clear"] == "true" and row["video_key"] in ready_set
        ]

    attr_priors = fit_attr_priors(punch_rows)
    video_by_key = {row["video_key"]: row for row in videos}

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
        count_sample_rows = sample_rows if args.split == "test" else []
        counts_by_key[key] = estimate_count(config, video_by_key[key], train_rows, punch_rows, count_sample_rows)

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

    if args.split == "test":
        rows = fill_sample_rows(sample_rows, test_videos, selected_by_video, attr_priors)
    else:
        rows = build_rows(videos, selected_by_video, attr_priors)
    write_csv_rows(args.output, rows, SUBMISSION_COLUMNS)
    if args.split == "test":
        errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
        if errors:
            print("Generated submission is invalid:")
            for error in errors:
                print(f"  - {error}")
            return 1
    print(f"Wrote {args.output}")
    print(f"split={args.split}")
    print(
        "selected="
        + ",".join(
            f"{key}:{len(selected_by_video.get(key, []))}" for key in sorted(selected_by_video)
        )
    )
    if args.score:
        if args.split != "validation":
            print("--score is only available for --split validation", file=sys.stderr)
            return 2
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        print(
            f"score={score['macro_score']:.6f} "
            f"time={summary['time']:.6f} fp={summary['fp_penalty']:.6f}",
            flush=True,
        )
    return 0


def complete_track_keys(tracks_dir: Path, video_by_key: dict[str, dict[str, str]]) -> list[str]:
    keys = []
    for path in sorted(tracks_dir.glob("*.jsonl")):
        video = video_by_key.get(path.stem)
        if video is None:
            continue
        with path.open("r", encoding="utf-8") as fh:
            line_count = sum(1 for _ in fh)
        if line_count == int(video["frame_count"]):
            keys.append(path.stem)
    return keys


def build_rows(
    videos: list[dict[str, str]],
    selected_by_video: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for video in sorted(videos, key=lambda row: row["video_key"]):
        for candidate in sorted(selected_by_video.get(video["video_key"], []), key=lambda item: item.frame):
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": video["video_key"],
                    "frame": str(max(0, min(int(video["frame_count"]) - 1, candidate.frame))),
                    "fighter": candidate.fighter,
                    "punch_type": attrs["punch_type"],
                    "hand": candidate.hand,
                    "target": candidate.target,
                    "effectiveness": attrs["effectiveness"],
                    "clear": "true",
                }
            )
            row_id += 1
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
