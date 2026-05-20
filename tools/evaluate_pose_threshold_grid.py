#!/usr/bin/env python3
"""Evaluate cached pose-candidate threshold/NMS/count policies.

This script caches pose candidates once per validation video and then runs a
grid quickly. It is meant to replace ad hoc one-off notebooks/commands.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--thresholds", default="0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.15,1.25,1.5")
    parser.add_argument("--nms-frames", default="4,6,8,10,12,16")
    parser.add_argument("--count-modes", default="threshold,root_rate,dataset_rate,root_round_rate,root_count")
    parser.add_argument("--count-multipliers", default="0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}

    ready_keys = []
    for path in sorted(args.tracks_dir.glob("*.jsonl")):
        key = path.stem
        if key not in video_by_key:
            continue
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count == int(video_by_key[key]["frame_count"]):
            ready_keys.append(key)
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

    ready_set = set(ready_keys)
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    train_punches = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    val_gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    attr_priors = fit_attr_priors(train_punches)
    train_counts = Counter(row["video_key"] for row in train_punches)

    print(f"ready={len(ready_keys)} {','.join(ready_keys)}")
    candidates_by_key = {
        key: score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        for key in ready_keys
    }

    thresholds = [float(value) for value in args.thresholds.split(",") if value]
    nms_values = [int(value) for value in args.nms_frames.split(",") if value]
    count_modes = [value for value in args.count_modes.split(",") if value]
    multipliers = [float(value) for value in args.count_multipliers.split(",") if value]

    results = []
    for mode in count_modes:
        mode_multipliers = [1.0] if mode == "threshold" else multipliers
        for multiplier in mode_multipliers:
            for threshold in thresholds:
                for nms_frames in nms_values:
                    pred = build_rows(
                        ready_keys,
                        video_by_key,
                        candidates_by_key,
                        attr_priors,
                        train_videos,
                        train_counts,
                        threshold,
                        nms_frames,
                        mode,
                        multiplier,
                    )
                    score = score_predictions(val_gt, pred)["macro_score"]
                    results.append((score, len(pred), mode, multiplier, threshold, nms_frames))

    print("score,n_pred,count_mode,count_multiplier,threshold,nms_frames")
    for score, n_pred, mode, multiplier, threshold, nms_frames in sorted(
        results, reverse=True
    )[: args.top_k]:
        print(f"{score:.6f},{n_pred},{mode},{multiplier},{threshold},{nms_frames}")
    return 0


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    threshold: float,
    nms_frames: int,
    count_mode: str,
    count_multiplier: float,
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for video_key in sorted(ready_keys):
        video = video_by_key[video_key]
        count = estimate_count(video, train_videos, train_counts, count_mode, count_multiplier)
        selected = select_candidates(
            candidates_by_key[video_key],
            PoseHeuristicConfig(min_score=threshold, nms_frames=nms_frames),
            count,
        )
        for candidate in selected:
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": video_key,
                    "frame": str(candidate.frame),
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


def estimate_count(
    video: dict[str, str],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    mode: str,
    multiplier: float,
) -> int | None:
    if mode == "threshold":
        return None
    if mode == "root_rate":
        peers = [row for row in train_videos if row["data_root"] == video["data_root"]] or train_videos
    elif mode == "dataset_rate":
        peers = [
            row for row in train_videos if row["dataset_type"] == video["dataset_type"]
        ] or train_videos
    elif mode == "root_round_rate":
        peers = [
            row
            for row in train_videos
            if row["data_root"] == video["data_root"]
            and row["round_number"] == video["round_number"]
        ]
        peers = peers or [row for row in train_videos if row["data_root"] == video["data_root"]]
        peers = peers or train_videos
    elif mode == "root_count":
        peers = [row for row in train_videos if row["data_root"] == video["data_root"]] or train_videos
        return round(sum(train_counts[row["video_key"]] for row in peers) / len(peers) * multiplier)
    else:
        raise ValueError(f"Unknown count mode: {mode}")

    n_punches = sum(train_counts[row["video_key"]] for row in peers)
    n_seconds = sum(int(row["frame_count"]) / 30.0 for row in peers)
    return round(n_punches / max(1e-9, n_seconds) * int(video["frame_count"]) / 30.0 * multiplier)


if __name__ == "__main__":
    raise SystemExit(main())
