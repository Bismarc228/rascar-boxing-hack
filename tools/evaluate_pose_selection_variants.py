#!/usr/bin/env python3
"""Evaluate NMS/grouping variants for pose punch candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
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
    parser.add_argument("--thresholds", default="0.6,0.7,0.8,0.9,1.0,1.15,1.3")
    parser.add_argument("--same-nms-frames", default="4,6,8,12")
    parser.add_argument("--cross-nms-frames", default="0,2,4,6,8")
    parser.add_argument(
        "--group-modes",
        default="global,fighter,fighter_hand,fighter_target,fighter_hand_target",
    )
    parser.add_argument(
        "--count-modes",
        default="threshold,root_rate,dataset_rate,root_round_rate,root_count,root_round_count,oracle",
    )
    parser.add_argument("--count-multipliers", default="0.6,0.7,0.8,0.9,1.0,1.1,1.25")
    parser.add_argument("--frame-offsets", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = [
        path.stem
        for path in sorted(args.tracks_dir.glob("*.jsonl"))
        if path.stem in video_by_key
        and sum(1 for _ in path.open("r", encoding="utf-8")) == int(video_by_key[path.stem]["frame_count"])
    ]
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

    ready_set = set(ready_keys)
    gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    train_gt = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    train_counts = Counter(row["video_key"] for row in train_gt)
    gt_counts = Counter(row["video_key"] for row in gt)
    attr_priors = fit_attr_priors(train_gt)
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}")

    candidates_by_key = {
        key: score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        for key in ready_keys
    }

    thresholds = parse_floats(args.thresholds)
    same_nms_values = parse_ints(args.same_nms_frames)
    cross_nms_values = parse_ints(args.cross_nms_frames)
    group_modes = [value for value in args.group_modes.split(",") if value]
    count_modes = [value for value in args.count_modes.split(",") if value]
    multipliers = parse_floats(args.count_multipliers)
    frame_offsets = parse_ints(args.frame_offsets)

    baseline_rows = build_rows(
        ready_keys,
        video_by_key,
        candidates_by_key,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        threshold=0.8,
        group_mode="global",
        same_nms=6,
        cross_nms=6,
        count_mode="threshold",
        count_multiplier=1.0,
        frame_offset=0,
    )
    baseline_score = score_predictions(gt, baseline_rows)
    print_score("baseline_global_thr08_nms6", baseline_score, len(baseline_rows), 0)

    results = []
    for group_mode in group_modes:
        for same_nms in same_nms_values:
            for cross_nms in cross_nms_values:
                if group_mode == "global" and same_nms != cross_nms:
                    continue
                for threshold in thresholds:
                    for count_mode in count_modes:
                        mode_multipliers = [1.0] if count_mode in {"threshold", "oracle"} else multipliers
                        for count_multiplier in mode_multipliers:
                            for frame_offset in frame_offsets:
                                rows = build_rows(
                                    ready_keys,
                                    video_by_key,
                                    candidates_by_key,
                                    attr_priors,
                                    train_videos,
                                    train_counts,
                                    gt_counts,
                                    threshold,
                                    group_mode,
                                    same_nms,
                                    cross_nms,
                                    count_mode,
                                    count_multiplier,
                                    frame_offset,
                                )
                                score = score_predictions(gt, rows)
                                summary = score_summary(score)
                                wins = video_wins(score, baseline_score)
                                results.append(
                                    (
                                        score["macro_score"],
                                        summary["time"],
                                        summary["fp_penalty"],
                                        wins,
                                        len(rows),
                                        group_mode,
                                        threshold,
                                        same_nms,
                                        cross_nms,
                                        count_mode,
                                        count_multiplier,
                                        frame_offset,
                                    )
                                )

    print(
        "score,time,fp_penalty,wins,n_pred,group_mode,threshold,same_nms,"
        "cross_nms,count_mode,count_multiplier,frame_offset"
    )
    for result in sorted(results, reverse=True)[: args.top_k]:
        (
            score,
            time_score,
            fp_penalty,
            wins,
            n_pred,
            group_mode,
            threshold,
            same_nms,
            cross_nms,
            count_mode,
            count_multiplier,
            frame_offset,
        ) = result
        print(
            f"{score:.6f},{time_score:.6f},{fp_penalty:.6f},{wins},{n_pred},"
            f"{group_mode},{threshold},{same_nms},{cross_nms},{count_mode},{count_multiplier},{frame_offset}"
        )
    return 0


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    threshold: float,
    group_mode: str,
    same_nms: int,
    cross_nms: int,
    count_mode: str,
    count_multiplier: float,
    frame_offset: int,
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        selected = select_candidates(
            candidates_by_key[key],
            PoseHeuristicConfig(
                min_score=threshold,
                nms_frames=same_nms,
                nms_group_mode=group_mode,
                cross_nms_frames=cross_nms,
            ),
            estimate_count(video, train_videos, train_counts, gt_counts, count_mode, count_multiplier),
        )
        for candidate in selected:
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": key,
                    "frame": str(
                        max(0, min(int(video["frame_count"]) - 1, candidate.frame + frame_offset))
                    ),
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
    gt_counts: Counter[str],
    mode: str,
    multiplier: float,
) -> int | None:
    if mode == "threshold":
        return None
    if mode == "oracle":
        return gt_counts[video["video_key"]]

    if mode == "root_rate":
        peers = [row for row in train_videos if row["data_root"] == video["data_root"]] or train_videos
        raw = rate_count(video, peers, train_counts)
    elif mode == "dataset_rate":
        peers = [
            row for row in train_videos if row["dataset_type"] == video["dataset_type"]
        ] or train_videos
        raw = rate_count(video, peers, train_counts)
    elif mode == "root_round_rate":
        peers = [
            row
            for row in train_videos
            if row["data_root"] == video["data_root"]
            and row["round_number"] == video["round_number"]
        ]
        peers = peers or [row for row in train_videos if row["data_root"] == video["data_root"]]
        peers = peers or train_videos
        raw = rate_count(video, peers, train_counts)
    elif mode == "root_count":
        peers = [row for row in train_videos if row["data_root"] == video["data_root"]] or train_videos
        raw = mean_count(peers, train_counts)
    elif mode == "root_round_count":
        peers = [
            row
            for row in train_videos
            if row["data_root"] == video["data_root"]
            and row["round_number"] == video["round_number"]
        ]
        peers = peers or [row for row in train_videos if row["data_root"] == video["data_root"]]
        peers = peers or train_videos
        raw = mean_count(peers, train_counts)
    else:
        raise ValueError(f"Unknown count mode: {mode}")
    return max(0, round(raw * multiplier))


def rate_count(
    video: dict[str, str],
    peers: list[dict[str, str]],
    train_counts: Counter[str],
) -> float:
    n_punches = sum(train_counts[row["video_key"]] for row in peers)
    n_seconds = sum(int(row["frame_count"]) / 30.0 for row in peers)
    return n_punches / max(1e-9, n_seconds) * int(video["frame_count"]) / 30.0


def mean_count(peers: list[dict[str, str]], train_counts: Counter[str]) -> float:
    return sum(train_counts[row["video_key"]] for row in peers) / max(1, len(peers))


def print_score(label: str, score: dict[str, object], n_pred: int, wins: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},time={summary['time']:.6f},"
        f"fp_penalty={summary['fp_penalty']:.6f},wins={wins},n_pred={n_pred}"
    )


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def video_wins(score: dict[str, object], baseline: dict[str, object], eps: float = 1e-9) -> int:
    return sum(
        int(score["by_video"][key]["final_score"] > baseline["by_video"][key]["final_score"] + eps)
        for key in baseline["by_video"]
    )


def parse_floats(values: str) -> list[float]:
    return [float(value) for value in values.split(",") if value]


def parse_ints(values: str) -> list[int]:
    return [int(value) for value in values.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
