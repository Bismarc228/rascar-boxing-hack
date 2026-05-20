#!/usr/bin/env python3
"""Evaluate cross-model pose-candidate agreement on cached validation tracks."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.ndimage import maximum_filter1d

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
from tools.evaluate_pose_selection_variants import (
    estimate_count,
    score_summary,
    video_wins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--primary-tracks-dir", type=Path, required=True)
    parser.add_argument("--secondary-tracks-dir", type=Path, required=True)
    parser.add_argument("--primary-name", default="primary")
    parser.add_argument("--secondary-name", default="secondary")
    parser.add_argument("--max-candidates-per-source-video", type=int, default=4000)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--windows", default="2,4,6")
    parser.add_argument("--agreement-alphas", default="0.0,0.2,0.5,1.0")
    parser.add_argument("--primary-weights", default="1.0,1.2")
    parser.add_argument("--secondary-weights", default="0.6,0.8,1.0")
    parser.add_argument("--thresholds", default="0.4,0.6,0.8,1.0,1.2,1.4,1.6")
    parser.add_argument("--nms-frames", default="8,10,12")
    parser.add_argument("--cross-nms-frames", default="4")
    parser.add_argument("--count-modes", default="threshold,root_count")
    parser.add_argument("--count-multipliers", default="0.8,0.9,1.0")
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    primary_keys = set(complete_track_keys(args.primary_tracks_dir, video_by_key))
    secondary_keys = set(complete_track_keys(args.secondary_tracks_dir, video_by_key))
    ready_keys = sorted(primary_keys & secondary_keys)
    if not ready_keys:
        print("No common complete validation tracks found")
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
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}", flush=True)

    primary_by_key = {
        key: normalized_candidates(
            args.primary_tracks_dir / f"{key}.jsonl",
            args.primary_name,
            args.pool_min_score,
            args.pool_nms_frames,
            args.max_candidates_per_source_video,
        )
        for key in ready_keys
    }
    secondary_by_key = {
        key: normalized_candidates(
            args.secondary_tracks_dir / f"{key}.jsonl",
            args.secondary_name,
            args.pool_min_score,
            args.pool_nms_frames,
            args.max_candidates_per_source_video,
        )
        for key in ready_keys
    }
    print(
        "pool="
        + ",".join(
            f"{key}:{len(primary_by_key[key])}+{len(secondary_by_key[key])}" for key in ready_keys
        ),
        flush=True,
    )

    baseline_rows = build_rows(
        ready_keys,
        video_by_key,
        primary_by_key,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        threshold=1.0,
        nms_frames=10,
        cross_nms=4,
        count_mode="root_count",
        count_multiplier=1.0,
    )
    baseline_score = score_predictions(gt, baseline_rows)
    print_score("primary_norm_ref", baseline_score, len(baseline_rows), 0)

    windows = parse_ints(args.windows)
    alphas = parse_floats(args.agreement_alphas)
    primary_weights = parse_floats(args.primary_weights)
    secondary_weights = parse_floats(args.secondary_weights)
    thresholds = parse_floats(args.thresholds)
    nms_values = parse_ints(args.nms_frames)
    cross_values = parse_ints(args.cross_nms_frames)
    count_modes = [value for value in args.count_modes.split(",") if value]
    multipliers = parse_floats(args.count_multipliers)

    results = []
    for window in windows:
        agreement_cache = {
            key: agreement_arrays(primary_by_key[key], secondary_by_key[key], window)
            for key in ready_keys
        }
        for alpha in alphas:
            for primary_weight in primary_weights:
                for secondary_weight in secondary_weights:
                    combined_by_key = {
                        key: combine_candidates(
                            primary_by_key[key],
                            secondary_by_key[key],
                            agreement_cache[key],
                            alpha,
                            primary_weight,
                            secondary_weight,
                        )
                        for key in ready_keys
                    }
                    for threshold in thresholds:
                        for nms_frames in nms_values:
                            for cross_nms in cross_values:
                                for count_mode in count_modes:
                                    mode_multipliers = (
                                        [1.0] if count_mode == "threshold" else multipliers
                                    )
                                    for count_multiplier in mode_multipliers:
                                        rows = build_rows(
                                            ready_keys,
                                            video_by_key,
                                            combined_by_key,
                                            attr_priors,
                                            train_videos,
                                            train_counts,
                                            gt_counts,
                                            threshold,
                                            nms_frames,
                                            cross_nms,
                                            count_mode,
                                            count_multiplier,
                                        )
                                        score = score_predictions(gt, rows)
                                        summary = score_summary(score)
                                        results.append(
                                            (
                                                score["macro_score"],
                                                summary["time"],
                                                summary["fp_penalty"],
                                                video_wins(score, baseline_score),
                                                len(rows),
                                                window,
                                                alpha,
                                                primary_weight,
                                                secondary_weight,
                                                threshold,
                                                nms_frames,
                                                cross_nms,
                                                count_mode,
                                                count_multiplier,
                                            )
                                        )

    print(
        "score,time,fp_penalty,wins,n_pred,window,alpha,primary_weight,"
        "secondary_weight,threshold,nms,cross_nms,count_mode,count_multiplier"
    )
    for result in sorted(results, reverse=True)[: args.top_k]:
        print(
            f"{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},{result[3]},"
            f"{result[4]},{result[5]},{result[6]},{result[7]},{result[8]},"
            f"{result[9]},{result[10]},{result[11]},{result[12]},{result[13]}"
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


def normalized_candidates(
    tracks_path: Path,
    source: str,
    pool_min_score: float,
    pool_nms_frames: int,
    max_candidates: int,
) -> list[PunchCandidate]:
    raw_candidates = score_pose_tracks(tracks_path, PoseHeuristicConfig(min_score=0.0))
    pooled = select_candidates(
        raw_candidates,
        PoseHeuristicConfig(
            min_score=pool_min_score,
            nms_frames=pool_nms_frames,
            nms_group_mode="fighter_hand",
            cross_nms_frames=0,
        ),
        max_candidates,
    )
    raw_scores = np.array([candidate.score for candidate in pooled], dtype=np.float32)
    scale = float(np.percentile(raw_scores, 95)) if len(raw_scores) else 1.0
    scale = max(scale, 1e-6)
    output = []
    for candidate in pooled:
        features = dict(candidate.features)
        features["raw_score"] = float(candidate.score)
        features["norm_score"] = float(candidate.score / scale)
        features["source"] = 0.0 if source == "primary" else 1.0
        output.append(
            PunchCandidate(
                candidate.video_key,
                candidate.frame,
                candidate.fighter,
                candidate.hand,
                candidate.target,
                float(candidate.score / scale),
                features,
            )
        )
    return output


def agreement_arrays(
    primary: list[PunchCandidate],
    secondary: list[PunchCandidate],
    window: int,
) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    max_frame = max([candidate.frame for candidate in primary + secondary] or [0]) + window + 2
    output = {}
    for name, candidates in [("primary", primary), ("secondary", secondary)]:
        group_arrays: dict[tuple[str, str], np.ndarray] = {}
        for candidate in candidates:
            group = (candidate.fighter, candidate.hand)
            if group not in group_arrays:
                group_arrays[group] = np.zeros(max_frame + 1, dtype=np.float32)
            group_arrays[group][candidate.frame] = max(
                group_arrays[group][candidate.frame],
                float(candidate.score),
            )
        if window > 0:
            size = 2 * window + 1
            group_arrays = {
                group: maximum_filter1d(values, size=size, mode="constant")
                for group, values in group_arrays.items()
            }
        output[name] = group_arrays
    return output


def combine_candidates(
    primary: list[PunchCandidate],
    secondary: list[PunchCandidate],
    agreement: dict[str, dict[tuple[str, str], np.ndarray]],
    alpha: float,
    primary_weight: float,
    secondary_weight: float,
) -> list[PunchCandidate]:
    output = []
    for source_name, other_name, weight, candidates in [
        ("primary", "secondary", primary_weight, primary),
        ("secondary", "primary", secondary_weight, secondary),
    ]:
        other_arrays = agreement[other_name]
        for candidate in candidates:
            group = (candidate.fighter, candidate.hand)
            other_score = 0.0
            if group in other_arrays and candidate.frame < len(other_arrays[group]):
                other_score = float(other_arrays[group][candidate.frame])
            score = candidate.score * weight * (1.0 + alpha * np.log1p(other_score))
            features = dict(candidate.features)
            features["agreement"] = other_score
            features["source_weight"] = weight
            features["source_name"] = 0.0 if source_name == "primary" else 1.0
            output.append(
                PunchCandidate(
                    candidate.video_key,
                    candidate.frame,
                    candidate.fighter,
                    candidate.hand,
                    candidate.target,
                    float(score),
                    features,
                )
            )
    return output


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    threshold: float,
    nms_frames: int,
    cross_nms: int,
    count_mode: str,
    count_multiplier: float,
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        count = estimate_count(video, train_videos, train_counts, gt_counts, count_mode, count_multiplier)
        selected = select_candidates(
            candidates_by_key[key],
            PoseHeuristicConfig(
                min_score=threshold,
                nms_frames=nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=cross_nms,
            ),
            count,
        )
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
    return rows


def print_score(label: str, score: dict[str, object], n_pred: int, wins: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},time={summary['time']:.6f},"
        f"fp_penalty={summary['fp_penalty']:.6f},wins={wins},n_pred={n_pred}",
        flush=True,
    )


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
