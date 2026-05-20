#!/usr/bin/env python3
"""Audit a normalized pose-agreement candidate against the YOLO11s anchor."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, estimate_attrs, fit_attr_priors, select_candidates
from tools.audit_pose_candidate import (
    ANCHOR_CONFIG,
    ANCHOR_TRACKS_DIR,
    build_rows as build_pose_rows,
    complete_track_keys,
    print_risk_flags,
    print_root_deltas,
    print_summary,
    print_video_deltas,
)
from tools.evaluate_pose_model_agreement import (
    agreement_arrays,
    combine_candidates,
    normalized_candidates,
)
from tools.evaluate_pose_selection_variants import estimate_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a normalized pose-agreement candidate against the fixed YOLO11s anchor."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--primary-tracks-dir", type=Path, required=True)
    parser.add_argument("--secondary-tracks-dir", type=Path, required=True)
    parser.add_argument("--anchor-tracks-dir", type=Path, default=ANCHOR_TRACKS_DIR)
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
    parser.add_argument(
        "--count-mode",
        choices=[
            "threshold",
            "dataset_count",
            "dataset_rate",
            "root_count",
            "root_rate",
            "root_round_count",
            "root_round_rate",
            "oracle",
        ],
        default="root_count",
    )
    parser.add_argument("--count-multiplier", type=float, default=0.88)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}

    primary_keys = complete_track_keys(args.primary_tracks_dir, video_by_key)
    secondary_keys = complete_track_keys(args.secondary_tracks_dir, video_by_key)
    anchor_keys = complete_track_keys(args.anchor_tracks_dir, video_by_key)
    ready_keys = sorted(set(primary_keys) & set(secondary_keys) & set(anchor_keys))
    if not ready_keys:
        print("No complete validation tracks found in primary, secondary, and anchor dirs")
        return 2

    ready_set = set(ready_keys)
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    train_punches = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    val_gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    attr_priors = fit_attr_priors(train_punches)
    train_counts = Counter(row["video_key"] for row in train_punches)
    gt_counts = Counter(row["video_key"] for row in val_gt)

    print(f"ready={len(ready_keys)} {','.join(ready_keys)}")
    if (
        len(primary_keys) != len(ready_keys)
        or len(secondary_keys) != len(ready_keys)
        or len(anchor_keys) != len(ready_keys)
    ):
        print(
            "complete_intersection_note="
            f"primary_complete={len(primary_keys)},secondary_complete={len(secondary_keys)},"
            f"anchor_complete={len(anchor_keys)},intersection={len(ready_keys)}"
        )

    anchor_rows, anchor_selected = build_pose_rows(
        ready_keys,
        video_by_key,
        args.anchor_tracks_dir,
        ANCHOR_CONFIG,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
    )
    candidate_rows, candidate_selected = build_agreement_rows(
        args,
        ready_keys,
        video_by_key,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
    )

    anchor_score = score_predictions(val_gt, anchor_rows)
    candidate_score = score_predictions(val_gt, candidate_rows)

    print()
    print(
        "anchor_yolo11s: "
        f"tracks_dir={args.anchor_tracks_dir},threshold={ANCHOR_CONFIG.min_score},"
        f"nms_frames={ANCHOR_CONFIG.nms_frames},nms_group_mode={ANCHOR_CONFIG.nms_group_mode},"
        f"cross_nms_frames={ANCHOR_CONFIG.cross_nms_frames},count_mode={ANCHOR_CONFIG.count_mode},"
        f"count_multiplier={ANCHOR_CONFIG.count_multiplier}"
    )
    print(
        "candidate_agreement: "
        f"primary={args.primary_tracks_dir},secondary={args.secondary_tracks_dir},"
        f"window={args.window},agreement_alpha={args.agreement_alpha},"
        f"primary_weight={args.primary_weight},secondary_weight={args.secondary_weight},"
        f"threshold={args.threshold},nms_frames={args.nms_frames},"
        f"cross_nms_frames={args.cross_nms_frames},count_mode={args.count_mode},"
        f"count_multiplier={args.count_multiplier}"
    )
    print()
    print_summary(anchor_score, candidate_score, len(anchor_rows), len(candidate_rows))
    print_risk_flags(videos, ready_keys, anchor_score, candidate_score)
    print()
    print_root_deltas(videos, ready_keys, anchor_score, candidate_score, anchor_selected, candidate_selected)
    print()
    print_video_deltas(videos, ready_keys, anchor_score, candidate_score, anchor_selected, candidate_selected)
    return 0


def build_agreement_rows(
    args: argparse.Namespace,
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
) -> tuple[list[dict[str, str]], dict[str, list]]:
    rows = []
    row_id = 1
    selected_by_key = {}
    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode="fighter_hand",
        cross_nms_frames=args.cross_nms_frames,
    )
    for key in ready_keys:
        video = video_by_key[key]
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
            gt_counts,
            args.count_mode,
            args.count_multiplier,
        )
        selected = select_candidates(combined, config, count)
        selected_by_key[key] = selected
        for candidate in selected:
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": key,
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
    return rows, selected_by_key


if __name__ == "__main__":
    raise SystemExit(main())
