#!/usr/bin/env python3
"""Generate a submission from cached YOLO-pose tracks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.pose_heuristic import PoseHeuristicConfig, make_pose_heuristic_submission
from rascar_boxing.validation import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/pose_heuristic.csv"))
    parser.add_argument(
        "--count-mode",
        choices=["threshold", "sample_true", "sample_capacity_fraction", "dataset_count", "dataset_rate", "fixed"],
        default="sample_true",
    )
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    parser.add_argument("--capacity-fraction", type=float, default=1.0)
    parser.add_argument("--max-predictions", type=int)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--nms-frames", type=int, default=12)
    parser.add_argument(
        "--nms-group-mode",
        choices=["global", "fighter", "fighter_hand", "fighter_target", "fighter_hand_target"],
        default="global",
    )
    parser.add_argument("--cross-nms-frames", type=int)
    parser.add_argument("--velocity-lag", type=int, default=3)
    parser.add_argument("--kp-conf", type=float, default=0.35)
    parser.add_argument("--frame-offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PoseHeuristicConfig(
        kp_conf=args.kp_conf,
        velocity_lag=args.velocity_lag,
        min_score=args.min_score,
        nms_frames=args.nms_frames,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
        capacity_fraction=args.capacity_fraction,
        max_predictions=args.max_predictions,
        frame_offset=args.frame_offset,
        nms_group_mode=args.nms_group_mode,
        cross_nms_frames=args.cross_nms_frames,
    )
    selected = make_pose_heuristic_submission(args.data_root, args.tracks_dir, args.output, config)
    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1

    by_video: dict[str, int] = {}
    for candidate in selected:
        by_video[candidate.video_key] = by_video.get(candidate.video_key, 0) + 1
    print(f"Wrote {args.output}")
    print(f"config={config}")
    print("selected=" + ",".join(f"{key}:{by_video[key]}" for key in sorted(by_video)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
