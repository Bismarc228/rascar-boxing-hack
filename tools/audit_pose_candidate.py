#!/usr/bin/env python3
"""Audit a cached pose candidate against the YOLO11s public-transfer anchor.

This is a CPU-only validation helper. It uses completed validation JSONL tracks,
scores one candidate pose-selection config, and compares it with the fixed
YOLO11s anchor that has transferred best to public so far.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)


ANCHOR_TRACKS_DIR = Path("data/processed/pose_tracks/val_yolo11s_conf035")
ANCHOR_CONFIG = PoseHeuristicConfig(
    min_score=1.15,
    nms_frames=10,
    nms_group_mode="fighter_hand",
    cross_nms_frames=4,
    context_feature="none",
    context_window=4,
    context_alpha=0.0,
    count_mode="root_count",
    count_multiplier=1.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one validation pose candidate against the fixed YOLO11s anchor."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--anchor-tracks-dir", type=Path, default=ANCHOR_TRACKS_DIR)
    parser.add_argument("--threshold", type=float, default=1.15)
    parser.add_argument("--nms-frames", type=int, default=10)
    parser.add_argument("--nms-group-mode", default="fighter_hand")
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument(
        "--context-feature",
        choices=["none", "same_sum", "same_count", "all_sum", "dominance"],
        default="none",
    )
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=0.0)
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
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}

    candidate_keys = complete_track_keys(args.tracks_dir, video_by_key)
    anchor_keys = complete_track_keys(args.anchor_tracks_dir, video_by_key)
    ready_keys = sorted(set(candidate_keys) & set(anchor_keys))
    if not ready_keys:
        print("No complete validation tracks found in both candidate and anchor dirs")
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

    candidate_config = PoseHeuristicConfig(
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

    print(f"ready={len(ready_keys)} {','.join(ready_keys)}")
    if len(candidate_keys) != len(ready_keys) or len(anchor_keys) != len(ready_keys):
        print(
            "complete_intersection_note="
            f"candidate_complete={len(candidate_keys)},anchor_complete={len(anchor_keys)},"
            f"intersection={len(ready_keys)}"
        )

    anchor_rows, anchor_selected = build_rows(
        ready_keys,
        video_by_key,
        args.anchor_tracks_dir,
        ANCHOR_CONFIG,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
    )
    candidate_rows, candidate_selected = build_rows(
        ready_keys,
        video_by_key,
        args.tracks_dir,
        candidate_config,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
    )

    anchor_score = score_predictions(val_gt, anchor_rows)
    candidate_score = score_predictions(val_gt, candidate_rows)

    print()
    print_config("anchor_yolo11s", args.anchor_tracks_dir, ANCHOR_CONFIG)
    print_config("candidate", args.tracks_dir, candidate_config)
    print()
    print_summary(anchor_score, candidate_score, len(anchor_rows), len(candidate_rows))
    print_risk_flags(videos, ready_keys, anchor_score, candidate_score)
    print()
    print_root_deltas(videos, ready_keys, anchor_score, candidate_score, anchor_selected, candidate_selected)
    print()
    print_video_deltas(videos, ready_keys, anchor_score, candidate_score, anchor_selected, candidate_selected)
    return 0


def complete_track_keys(tracks_dir: Path, video_by_key: dict[str, dict[str, str]]) -> list[str]:
    keys = []
    for path in sorted(tracks_dir.glob("*.jsonl")):
        key = path.stem
        if key not in video_by_key:
            continue
        if count_lines(path) == int(video_by_key[key]["frame_count"]):
            keys.append(key)
    return keys


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    tracks_dir: Path,
    config: PoseHeuristicConfig,
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
) -> tuple[list[dict[str, str]], dict[str, list[PunchCandidate]]]:
    rows = []
    row_id = 1
    selected_by_key = {}
    for video_key in ready_keys:
        video = video_by_key[video_key]
        candidates = score_pose_tracks(tracks_dir / f"{video_key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        candidates = apply_temporal_context(candidates, config)
        count = estimate_count(config, video, train_videos, train_counts, gt_counts)
        selected = select_candidates(candidates, config, count)
        selected_by_key[video_key] = selected
        for candidate in selected:
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": video_key,
                    "frame": str(clamp_frame(candidate.frame, video)),
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


def estimate_count(
    config: PoseHeuristicConfig,
    video: dict[str, str],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
) -> int | None:
    if config.count_mode == "threshold":
        return None
    if config.count_mode == "oracle":
        return gt_counts[video["video_key"]]

    if config.count_mode.startswith("dataset_"):
        peers = [row for row in train_videos if row["dataset_type"] == video["dataset_type"]]
    elif config.count_mode.startswith("root_round_"):
        peers = [
            row
            for row in train_videos
            if row["data_root"] == video["data_root"]
            and row["round_number"] == video["round_number"]
        ]
        peers = peers or [row for row in train_videos if row["data_root"] == video["data_root"]]
    elif config.count_mode.startswith("root_"):
        peers = [row for row in train_videos if row["data_root"] == video["data_root"]]
    else:
        raise ValueError(f"Unknown count mode: {config.count_mode}")
    peers = peers or train_videos

    if config.count_mode.endswith("_count"):
        raw_count = sum(train_counts[row["video_key"]] for row in peers) / max(1, len(peers))
    else:
        n_punches = sum(train_counts[row["video_key"]] for row in peers)
        n_seconds = sum(int(row["frame_count"]) / 30.0 for row in peers)
        raw_count = n_punches / max(1e-9, n_seconds) * int(video["frame_count"]) / 30.0
    return max(0, round(raw_count * config.count_multiplier))


def clamp_frame(frame: int, video: dict[str, str]) -> int:
    return max(0, min(int(video["frame_count"]) - 1, frame))


def print_config(label: str, tracks_dir: Path, config: PoseHeuristicConfig) -> None:
    print(
        f"{label}: tracks_dir={tracks_dir},threshold={config.min_score},"
        f"nms_frames={config.nms_frames},nms_group_mode={config.nms_group_mode},"
        f"cross_nms_frames={config.cross_nms_frames},context_feature={config.context_feature},"
        f"context_window={config.context_window},context_alpha={config.context_alpha},"
        f"count_mode={config.count_mode},count_multiplier={config.count_multiplier}"
    )


def print_summary(
    anchor_score: dict[str, Any],
    candidate_score: dict[str, Any],
    anchor_n_pred: int,
    candidate_n_pred: int,
) -> None:
    anchor = score_summary(anchor_score)
    candidate = score_summary(candidate_score)
    wins = video_wins(candidate_score, anchor_score)
    losses = video_wins(anchor_score, candidate_score)
    ties = len(anchor_score["by_video"]) - wins - losses
    print("summary")
    print(
        "label,macro,time,fp_penalty,n_pred,wins_vs_anchor,losses_vs_anchor,"
        "ties_vs_anchor,delta_macro,delta_time,delta_fp_penalty,delta_n_pred"
    )
    print(
        f"anchor_yolo11s,{anchor_score['macro_score']:.6f},{anchor['time']:.6f},"
        f"{anchor['fp_penalty']:.6f},{anchor_n_pred},0,0,0,0.000000,0.000000,0.000000,0"
    )
    print(
        f"candidate,{candidate_score['macro_score']:.6f},{candidate['time']:.6f},"
        f"{candidate['fp_penalty']:.6f},{candidate_n_pred},{wins},{losses},{ties},"
        f"{candidate_score['macro_score'] - anchor_score['macro_score']:.6f},"
        f"{candidate['time'] - anchor['time']:.6f},"
        f"{candidate['fp_penalty'] - anchor['fp_penalty']:.6f},"
        f"{candidate_n_pred - anchor_n_pred}"
    )


def print_risk_flags(
    videos: list[dict[str, str]],
    ready_keys: list[str],
    anchor_score: dict[str, Any],
    candidate_score: dict[str, Any],
) -> None:
    video_by_key = {row["video_key"]: row for row in videos}
    root_keys: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in ready_keys:
        video = video_by_key[key]
        root_keys[(video["dataset_type"], video["data_root"])].append(key)

    tournament_drops = []
    for (dataset_type, data_root), keys in sorted(root_keys.items()):
        if not dataset_type.startswith("tournament"):
            continue
        anchor = aggregate_score(anchor_score, keys)
        candidate = aggregate_score(candidate_score, keys)
        delta = candidate["macro"] - anchor["macro"]
        if delta < -1e-9:
            tournament_drops.append(f"{dataset_type}|{data_root}:{delta:.6f}")

    macro_delta = candidate_score["macro_score"] - anchor_score["macro_score"]
    if macro_delta > 1e-9 and tournament_drops:
        print(
            "risk_flags=macro_up_but_tournament_roots_down "
            f"delta_macro={macro_delta:.6f} roots={';'.join(tournament_drops)}"
        )
    elif tournament_drops:
        print(f"risk_flags=tournament_roots_down roots={';'.join(tournament_drops)}")
    else:
        print("risk_flags=none")


def print_root_deltas(
    videos: list[dict[str, str]],
    ready_keys: list[str],
    anchor_score: dict[str, Any],
    candidate_score: dict[str, Any],
    anchor_selected: dict[str, list[PunchCandidate]],
    candidate_selected: dict[str, list[PunchCandidate]],
) -> None:
    video_by_key = {row["video_key"]: row for row in videos}
    roots: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in ready_keys:
        video = video_by_key[key]
        roots[(video["dataset_type"], video["data_root"])].append(key)

    print("root_deltas")
    print(
        "dataset_type,data_root,n_videos,anchor_macro,candidate_macro,delta_macro,"
        "anchor_time,candidate_time,delta_time,anchor_fp,candidate_fp,delta_fp,"
        "anchor_n_pred,candidate_n_pred,delta_n_pred,wins,losses"
    )
    for (dataset_type, data_root), keys in sorted(roots.items()):
        anchor = aggregate_score(anchor_score, keys)
        candidate = aggregate_score(candidate_score, keys)
        anchor_n = sum(len(anchor_selected[key]) for key in keys)
        candidate_n = sum(len(candidate_selected[key]) for key in keys)
        wins = sum(
            candidate_score["by_video"][key]["final_score"] > anchor_score["by_video"][key]["final_score"] + 1e-9
            for key in keys
        )
        losses = sum(
            anchor_score["by_video"][key]["final_score"] > candidate_score["by_video"][key]["final_score"] + 1e-9
            for key in keys
        )
        print(
            f"{dataset_type},{data_root},{len(keys)},{anchor['macro']:.6f},{candidate['macro']:.6f},"
            f"{candidate['macro'] - anchor['macro']:.6f},{anchor['time']:.6f},{candidate['time']:.6f},"
            f"{candidate['time'] - anchor['time']:.6f},{anchor['fp_penalty']:.6f},"
            f"{candidate['fp_penalty']:.6f},{candidate['fp_penalty'] - anchor['fp_penalty']:.6f},"
            f"{anchor_n},{candidate_n},{candidate_n - anchor_n},{wins},{losses}"
        )


def print_video_deltas(
    videos: list[dict[str, str]],
    ready_keys: list[str],
    anchor_score: dict[str, Any],
    candidate_score: dict[str, Any],
    anchor_selected: dict[str, list[PunchCandidate]],
    candidate_selected: dict[str, list[PunchCandidate]],
) -> None:
    video_by_key = {row["video_key"]: row for row in videos}
    print("video_deltas")
    print(
        "video_key,dataset_type,data_root,round_number,anchor_score,candidate_score,delta_score,"
        "anchor_time,candidate_time,delta_time,anchor_fp,candidate_fp,delta_fp,"
        "gt_count,anchor_n_pred,candidate_n_pred,delta_n_pred,result"
    )
    for key in ready_keys:
        video = video_by_key[key]
        anchor = anchor_score["by_video"][key]
        candidate = candidate_score["by_video"][key]
        delta = candidate["final_score"] - anchor["final_score"]
        if delta > 1e-9:
            result = "win"
        elif delta < -1e-9:
            result = "loss"
        else:
            result = "tie"
        anchor_n = len(anchor_selected[key])
        candidate_n = len(candidate_selected[key])
        print(
            f"{key},{video['dataset_type']},{video['data_root']},{video['round_number']},"
            f"{anchor['final_score']:.6f},{candidate['final_score']:.6f},{delta:.6f},"
            f"{anchor['score_time']:.6f},{candidate['score_time']:.6f},"
            f"{candidate['score_time'] - anchor['score_time']:.6f},"
            f"{anchor['fp_penalty']:.6f},{candidate['fp_penalty']:.6f},"
            f"{candidate['fp_penalty'] - anchor['fp_penalty']:.6f},"
            f"{anchor['n_gt']},{anchor_n},{candidate_n},{candidate_n - anchor_n},{result}"
        )


def score_summary(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def aggregate_score(score: dict[str, Any], keys: list[str]) -> dict[str, float]:
    values = [score["by_video"][key] for key in keys]
    return {
        "macro": float(np.mean([item["final_score"] for item in values])),
        "time": float(np.mean([item["score_time"] for item in values])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in values])),
    }


def video_wins(score: dict[str, Any], baseline: dict[str, Any], eps: float = 1e-9) -> int:
    return sum(
        int(score["by_video"][key]["final_score"] > baseline["by_video"][key]["final_score"] + eps)
        for key in baseline["by_video"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
