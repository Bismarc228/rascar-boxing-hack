#!/usr/bin/env python3
"""Write validation prediction rows from cached pose tracks for one config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, predict_for_videos
from tools.evaluate_pose_selection_variants import score_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-keys", help="Optional comma-separated validation keys")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--nms-frames", type=int, required=True)
    parser.add_argument("--nms-group-mode", default="fighter_hand")
    parser.add_argument("--cross-nms-frames", type=int, default=0)
    parser.add_argument("--context-feature", default="none")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=0.0)
    parser.add_argument("--count-mode", default="threshold")
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    parser.add_argument("--score", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if args.video_keys:
        requested = {value for value in args.video_keys.split(",") if value}
        ready_keys = [key for key in ready_keys if key in requested]
    if not ready_keys:
        print("No complete validation tracks found", file=sys.stderr)
        return 2

    ready_set = set(ready_keys)
    val_videos = [row for row in videos if row["video_key"] in ready_set]
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    train_punches = [row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set]
    val_gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode=args.nms_group_mode,
        cross_nms_frames=args.cross_nms_frames,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
    )
    rows = predict_for_videos(
        val_videos,
        args.tracks_dir,
        config,
        train_videos=train_videos,
        train_punches=train_punches,
    )
    write_csv_rows(args.output, rows, SUBMISSION_COLUMNS)
    print(f"wrote={args.output}")
    print(f"ready={len(ready_keys)} {','.join(sorted(ready_keys))}")
    print(f"n_rows={len(rows)}")
    if args.score:
        score = score_predictions(val_gt, rows)
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
        key = path.stem
        if key not in video_by_key:
            continue
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count == int(video_by_key[key]["frame_count"]):
            keys.append(key)
    return keys


if __name__ == "__main__":
    raise SystemExit(main())
