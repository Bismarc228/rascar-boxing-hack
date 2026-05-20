#!/usr/bin/env python3
"""Evaluate temporal-context rescoring of pose punch candidates."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--pool-threshold", type=float, default=0.0)
    parser.add_argument("--thresholds", default="0.6,0.7,0.8,0.9,1.0,1.15")
    parser.add_argument("--nms-frames", default="4,6,8")
    parser.add_argument("--nms-group-modes", default="global,fighter")
    parser.add_argument("--cross-nms-frames", default="2,4,6")
    parser.add_argument("--windows", default="2,4,6,8,12")
    parser.add_argument("--alphas", default="-0.4,-0.2,0.0,0.1,0.2,0.35,0.5")
    parser.add_argument("--features", default="same_sum,same_count,all_sum,dominance")
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
    attr_priors = fit_attr_priors(train_gt)
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}", flush=True)

    candidates_by_key = {
        key: [
            candidate
            for candidate in score_pose_tracks(
                args.tracks_dir / f"{key}.jsonl",
                PoseHeuristicConfig(min_score=0.0),
            )
            if candidate.score >= args.pool_threshold
        ]
        for key in ready_keys
    }

    baseline_by_key = {
        key: select_candidates(
            candidates_by_key[key],
            PoseHeuristicConfig(min_score=0.8, nms_frames=6),
            None,
        )
        for key in ready_keys
    }
    baseline_rows = rows_from_candidates(ready_keys, video_by_key, baseline_by_key, attr_priors)
    baseline_score = score_predictions(gt, baseline_rows)
    print_score("baseline_global_thr08_nms6", baseline_score, len(baseline_rows), 0)

    thresholds = parse_floats(args.thresholds)
    nms_values = parse_ints(args.nms_frames)
    group_modes = [value for value in args.nms_group_modes.split(",") if value]
    cross_values = parse_ints(args.cross_nms_frames)
    windows = parse_ints(args.windows)
    alphas = parse_floats(args.alphas)
    feature_names = [value for value in args.features.split(",") if value]

    results = []
    for window in windows:
        context_by_key = {
            key: add_context_scores(candidates, window)
            for key, candidates in candidates_by_key.items()
        }
        for feature_name in feature_names:
            for alpha in alphas:
                rescored_by_key = {
                    key: [
                        rescore(candidate, feature_name, alpha)
                        for candidate in candidates
                    ]
                    for key, candidates in context_by_key.items()
                }
                for group_mode in group_modes:
                    for nms_frames in nms_values:
                        for cross_nms in cross_values:
                            if group_mode == "global" and nms_frames != cross_nms:
                                continue
                            for threshold in thresholds:
                                selected_by_key = {
                                    key: select_candidates(
                                        rescored_by_key[key],
                                        PoseHeuristicConfig(
                                            min_score=threshold,
                                            nms_frames=nms_frames,
                                            nms_group_mode=group_mode,
                                            cross_nms_frames=cross_nms,
                                        ),
                                        None,
                                    )
                                    for key in ready_keys
                                }
                                rows = rows_from_candidates(
                                    ready_keys,
                                    video_by_key,
                                    selected_by_key,
                                    attr_priors,
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
                                        feature_name,
                                        window,
                                        alpha,
                                        group_mode,
                                        threshold,
                                        nms_frames,
                                        cross_nms,
                                    )
                                )

    print("score,time,fp_penalty,wins,n_pred,feature,window,alpha,group_mode,threshold,nms,cross_nms")
    for result in sorted(results, reverse=True)[: args.top_k]:
        (
            score,
            time_score,
            fp_penalty,
            wins,
            n_pred,
            feature_name,
            window,
            alpha,
            group_mode,
            threshold,
            nms_frames,
            cross_nms,
        ) = result
        print(
            f"{score:.6f},{time_score:.6f},{fp_penalty:.6f},{wins},{n_pred},"
            f"{feature_name},{window},{alpha},{group_mode},{threshold},{nms_frames},{cross_nms}"
        )
    return 0


def add_context_scores(candidates: list[PunchCandidate], window: int) -> list[PunchCandidate]:
    if not candidates:
        return []
    max_frame = max(candidate.frame for candidate in candidates) + window + 2
    all_scores = np.zeros(max_frame + 1, dtype=np.float32)
    all_counts = np.zeros(max_frame + 1, dtype=np.float32)
    group_scores: dict[tuple[str, str], np.ndarray] = {}
    group_counts: dict[tuple[str, str], np.ndarray] = {}
    for candidate in candidates:
        frame = max(0, min(max_frame, candidate.frame))
        group = (candidate.fighter, candidate.hand)
        all_scores[frame] += candidate.score
        all_counts[frame] += 1.0
        if group not in group_scores:
            group_scores[group] = np.zeros(max_frame + 1, dtype=np.float32)
            group_counts[group] = np.zeros(max_frame + 1, dtype=np.float32)
        group_scores[group][frame] += candidate.score
        group_counts[group][frame] += 1.0

    all_scores_prefix = np.concatenate([[0.0], np.cumsum(all_scores)])
    all_counts_prefix = np.concatenate([[0.0], np.cumsum(all_counts)])
    group_score_prefix = {
        key: np.concatenate([[0.0], np.cumsum(values)])
        for key, values in group_scores.items()
    }
    group_count_prefix = {
        key: np.concatenate([[0.0], np.cumsum(values)])
        for key, values in group_counts.items()
    }

    output = []
    for candidate in candidates:
        lo = max(0, candidate.frame - window)
        hi = min(max_frame, candidate.frame + window) + 1
        group = (candidate.fighter, candidate.hand)
        all_sum = float(all_scores_prefix[hi] - all_scores_prefix[lo])
        same_sum = float(group_score_prefix[group][hi] - group_score_prefix[group][lo])
        same_count = float(group_count_prefix[group][hi] - group_count_prefix[group][lo])
        other_sum = max(0.0, all_sum - same_sum)
        dominance = same_sum / max(1e-6, other_sum)
        features = dict(candidate.features)
        features["ctx_same_sum"] = same_sum
        features["ctx_same_count"] = same_count
        features["ctx_all_sum"] = all_sum
        features["ctx_dominance"] = dominance
        output.append(
            PunchCandidate(
                candidate.video_key,
                candidate.frame,
                candidate.fighter,
                candidate.hand,
                candidate.target,
                candidate.score,
                features,
            )
        )
    return output


def rescore(candidate: PunchCandidate, feature_name: str, alpha: float) -> PunchCandidate:
    if alpha == 0.0:
        return candidate
    if feature_name == "same_sum":
        value = candidate.features.get("ctx_same_sum", 0.0) / max(1e-6, candidate.score)
    elif feature_name == "same_count":
        value = candidate.features.get("ctx_same_count", 0.0)
    elif feature_name == "all_sum":
        value = candidate.features.get("ctx_all_sum", 0.0) / max(1e-6, candidate.score)
    elif feature_name == "dominance":
        value = candidate.features.get("ctx_dominance", 0.0)
    else:
        raise ValueError(f"Unknown context feature: {feature_name}")
    score = candidate.score * max(0.01, 1.0 + alpha * np.log1p(value))
    return PunchCandidate(
        candidate.video_key,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
        float(score),
        candidate.features,
    )


def rows_from_candidates(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        for candidate in sorted(selected_by_key[key], key=lambda item: item.frame):
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
