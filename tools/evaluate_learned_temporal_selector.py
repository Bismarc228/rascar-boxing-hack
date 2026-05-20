#!/usr/bin/env python3
"""Evaluate a fight-group OOF learned selector over cached pose candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

from rascar_boxing.constants import FPS
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


BASE_FEATURES = [
    "closing",
    "arm_forward",
    "reach",
    "proximity",
    "conf",
    "role_conf",
    "attacker_red_score",
    "attacker_blue_score",
    "attacker_color_margin",
    "target_dist_norm",
    "head_dist_norm",
    "body_dist_norm",
    "forward_norm",
]


@dataclass(frozen=True)
class FeaturePack:
    candidates: list[PunchCandidate]
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["hgb", "lgbm"], default="hgb")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=4000)
    parser.add_argument("--label-window", type=int, default=15)
    parser.add_argument("--hard-negative-window", type=int, default=45)
    parser.add_argument("--windows", default="2,4,8,12")
    parser.add_argument("--thresholds", default="0.0,0.005,0.01,0.02,0.04,0.08,0.12,0.18,0.25")
    parser.add_argument("--nms-frames", default="8,10,12")
    parser.add_argument("--cross-nms-frames", default="2,4")
    parser.add_argument("--count-modes", default="threshold,root_count")
    parser.add_argument("--count-multipliers", default="0.8,0.9,1.0")
    parser.add_argument("--pose-priors", default="0.0,0.05,0.1")
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if len(ready_keys) < 3:
        print(f"Need at least 3 complete train tracks, got {ready_keys}")
        return 2

    ready_set = set(ready_keys)
    windows = parse_ints(args.windows)
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}", flush=True)

    gt_by_key: dict[str, list[dict[str, str]]] = {}
    clear_gt = [row for row in punches if row["clear"] == "true"]
    for row in clear_gt:
        gt_by_key.setdefault(row["video_key"], []).append(row)

    packs: dict[str, FeaturePack] = {}
    for key in ready_keys:
        candidates = score_pose_tracks(
            args.tracks_dir / f"{key}.jsonl",
            PoseHeuristicConfig(min_score=0.0),
        )
        candidates = select_candidates(
            candidates,
            PoseHeuristicConfig(
                min_score=args.pool_min_score,
                nms_frames=args.pool_nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=0,
            ),
            args.max_candidates_per_video,
        )
        x = build_features(candidates, video_by_key[key], windows)
        y, weight = build_labels(
            candidates,
            gt_by_key.get(key, []),
            args.label_window,
            args.hard_negative_window,
        )
        packs[key] = FeaturePack(candidates, x, y, weight)

    print("pool=" + ",".join(f"{key}:{len(packs[key].candidates)}" for key in ready_keys), flush=True)

    group_by_key = {key: fight_group(video_by_key[key]) for key in ready_keys}
    groups = sorted(set(group_by_key.values()))
    scored_by_key: dict[str, list[PunchCandidate]] = {}
    for group in groups:
        valid_keys = [key for key in ready_keys if group_by_key[key] == group]
        train_keys = [key for key in ready_keys if group_by_key[key] != group]
        x_train = np.concatenate([packs[key].x for key in train_keys], axis=0)
        y_train = np.concatenate([packs[key].y for key in train_keys], axis=0)
        w_train = np.concatenate([packs[key].weight for key in train_keys], axis=0)
        if len(np.unique(y_train)) < 2:
            print(f"skip_group={group} reason=single_class")
            continue

        model = fit_model(args.model, x_train, y_train, w_train, args.threads)
        for key in valid_keys:
            proba = model.predict_proba(packs[key].x)[:, 1]
            scored_by_key[key] = [
                replace_score(candidate, float(probability))
                for candidate, probability in zip(packs[key].candidates, proba)
            ]
        print(
            f"fold={group} train={len(train_keys)} valid={','.join(valid_keys)} "
            f"pos={int(y_train.sum())}/{len(y_train)}",
            flush=True,
        )

    if set(scored_by_key) != ready_set:
        missing = sorted(ready_set - set(scored_by_key))
        print(f"Missing OOF predictions for {missing}")
        return 2

    train_gt = [row for row in clear_gt if row["video_key"] not in ready_set]
    attr_priors = fit_attr_priors(train_gt)
    gt = [row for row in clear_gt if row["video_key"] in ready_set]
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    train_counts = Counter(row["video_key"] for row in train_gt)
    gt_counts = Counter(row["video_key"] for row in gt)

    results = []
    thresholds = parse_floats(args.thresholds)
    nms_values = parse_ints(args.nms_frames)
    cross_values = parse_ints(args.cross_nms_frames)
    count_modes = [value for value in args.count_modes.split(",") if value]
    multipliers = parse_floats(args.count_multipliers)
    pose_priors = parse_floats(args.pose_priors)

    for pose_prior in pose_priors:
        reranked = {
            key: [apply_pose_prior(candidate, pose_prior) for candidate in candidates]
            for key, candidates in scored_by_key.items()
        }
        for threshold in thresholds:
            for nms_frames in nms_values:
                for cross_nms in cross_values:
                    for count_mode in count_modes:
                        mode_multipliers = [1.0] if count_mode == "threshold" else multipliers
                        for count_multiplier in mode_multipliers:
                            rows = build_rows(
                                ready_keys,
                                video_by_key,
                                reranked,
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
                                    len(rows),
                                    pose_prior,
                                    threshold,
                                    nms_frames,
                                    cross_nms,
                                    count_mode,
                                    count_multiplier,
                                )
                            )

    print(
        "score,time,fp_penalty,n_pred,pose_prior,threshold,nms,cross_nms,"
        "count_mode,count_multiplier"
    )
    for result in sorted(results, reverse=True)[: args.top_k]:
        print(
            f"{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},{result[3]},"
            f"{result[4]},{result[5]},{result[6]},{result[7]},{result[8]},{result[9]}"
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


def build_features(
    candidates: list[PunchCandidate],
    video: dict[str, str],
    windows: list[int],
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, feature_count(windows)), dtype=np.float32)

    max_window = max(windows) if windows else 0
    max_frame = max(candidate.frame for candidate in candidates) + max_window + 2
    scores = np.zeros(max_frame + 1, dtype=np.float32)
    counts = np.zeros(max_frame + 1, dtype=np.float32)
    group_scores: dict[tuple[str, str], np.ndarray] = {}
    group_counts: dict[tuple[str, str], np.ndarray] = {}
    group_frames: dict[tuple[str, str], list[int]] = {}
    all_frames = []
    for candidate in candidates:
        frame = max(0, min(max_frame, candidate.frame))
        group = (candidate.fighter, candidate.hand)
        scores[frame] += candidate.score
        counts[frame] += 1.0
        all_frames.append(frame)
        group_frames.setdefault(group, []).append(frame)
        if group not in group_scores:
            group_scores[group] = np.zeros(max_frame + 1, dtype=np.float32)
            group_counts[group] = np.zeros(max_frame + 1, dtype=np.float32)
        group_scores[group][frame] += candidate.score
        group_counts[group][frame] += 1.0

    score_prefix = np.concatenate([[0.0], np.cumsum(scores)])
    count_prefix = np.concatenate([[0.0], np.cumsum(counts)])
    group_score_prefix = {
        key: np.concatenate([[0.0], np.cumsum(value)])
        for key, value in group_scores.items()
    }
    group_count_prefix = {
        key: np.concatenate([[0.0], np.cumsum(value)])
        for key, value in group_counts.items()
    }
    sorted_all_frames = np.array(sorted(all_frames), dtype=np.int32)
    sorted_group_frames = {
        key: np.array(sorted(value), dtype=np.int32) for key, value in group_frames.items()
    }

    raw_scores = np.array([candidate.score for candidate in candidates], dtype=np.float32)
    order = np.argsort(-raw_scores)
    ranks = np.empty(len(candidates), dtype=np.float32)
    ranks[order] = np.arange(len(candidates), dtype=np.float32)
    score_pct = 1.0 - ranks / max(1.0, len(candidates) - 1)

    rows = []
    frame_count = float(video["frame_count"])
    for index, candidate in enumerate(candidates):
        values = candidate.features
        group = (candidate.fighter, candidate.hand)
        row = [
            float(np.log1p(max(0.0, candidate.score))),
            float(candidate.score),
            float(score_pct[index]),
            float(candidate.frame / max(1.0, frame_count)),
            1.0 if candidate.fighter == "red" else 0.0,
            1.0 if candidate.hand == "left" else 0.0,
            1.0 if candidate.target == "head" else 0.0,
        ]
        for name in BASE_FEATURES:
            value = values.get(name, 0.0)
            if name == "forward_norm":
                value /= 1000.0
            row.append(float(value))

        for window in windows:
            lo = max(0, candidate.frame - window)
            hi = min(max_frame, candidate.frame + window) + 1
            all_sum = float(score_prefix[hi] - score_prefix[lo])
            all_count = float(count_prefix[hi] - count_prefix[lo])
            same_sum = float(group_score_prefix[group][hi] - group_score_prefix[group][lo])
            same_count = float(group_count_prefix[group][hi] - group_count_prefix[group][lo])
            other_sum = max(0.0, all_sum - same_sum)
            dominance = same_sum / max(1e-6, other_sum)
            same_share = same_sum / max(1e-6, all_sum)
            row.extend(
                [
                    all_sum,
                    all_count,
                    same_sum,
                    same_count,
                    other_sum,
                    dominance,
                    same_share,
                    nearest_gap(sorted_all_frames, candidate.frame, window),
                    nearest_gap(sorted_group_frames[group], candidate.frame, window),
                ]
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def feature_count(windows: list[int]) -> int:
    return 7 + len(BASE_FEATURES) + 9 * len(windows)


def nearest_gap(frames: np.ndarray, frame: int, cap: int) -> float:
    if len(frames) <= 1:
        return 1.0
    pos = int(np.searchsorted(frames, frame))
    best = cap + 1
    if pos > 0:
        gap = abs(frame - int(frames[pos - 1]))
        if gap > 0:
            best = min(best, gap)
    if pos + 1 < len(frames):
        gap = abs(frame - int(frames[pos + 1]))
        if gap > 0:
            best = min(best, gap)
    return float(min(best, cap + 1) / max(1, cap + 1))


def build_labels(
    candidates: list[PunchCandidate],
    gt_rows: list[dict[str, str]],
    label_window: int,
    hard_negative_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(len(candidates), dtype=np.int8)
    weight = np.ones(len(candidates), dtype=np.float32)
    gt_frames = np.array([int(row["frame"]) for row in gt_rows], dtype=np.int32)
    if len(gt_frames) == 0:
        return y, weight

    for index, candidate in enumerate(candidates):
        diffs = np.abs(gt_frames - candidate.frame)
        best_index = int(np.argmin(diffs))
        best = int(diffs[best_index])
        if best <= label_window:
            gt = gt_rows[best_index]
            y[index] = 1
            time_weight = max(0.25, 1.0 - best / max(1, label_window))
            attr_bonus = 1.0
            if candidate.fighter == gt["fighter"]:
                attr_bonus += 0.25
            if candidate.hand == gt["hand"]:
                attr_bonus += 0.15
            weight[index] = 2.0 * time_weight * attr_bonus
        elif best <= hard_negative_window:
            weight[index] = 2.0
    return y, weight


def fit_model(model_name: str, x: np.ndarray, y: np.ndarray, weight: np.ndarray, threads: int):
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=0.5,
            objective="binary",
            n_jobs=threads,
            random_state=42,
            verbosity=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        )
    model.fit(x, y, sample_weight=weight)
    return model


def apply_pose_prior(candidate: PunchCandidate, alpha: float) -> PunchCandidate:
    score = candidate.score * (1.0 + alpha * np.log1p(max(0.0, candidate.features.get("pose_score", 0.0))))
    return replace_score(candidate, float(score))


def replace_score(candidate: PunchCandidate, score: float) -> PunchCandidate:
    features = dict(candidate.features)
    features.setdefault("pose_score", candidate.score)
    return PunchCandidate(
        candidate.video_key,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
        score,
        features,
    )


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
        selected = select_candidates(
            candidates_by_key[key],
            PoseHeuristicConfig(
                min_score=threshold,
                nms_frames=nms_frames,
                nms_group_mode="fighter_hand",
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
    if mode == "root_count":
        peers = [row for row in train_videos if row["data_root"] == video["data_root"]] or train_videos
        raw = sum(train_counts[row["video_key"]] for row in peers) / max(1, len(peers))
    elif mode == "root_rate":
        peers = [row for row in train_videos if row["data_root"] == video["data_root"]] or train_videos
        raw = rate_count(video, peers, train_counts)
    else:
        raise ValueError(f"Unknown count_mode: {mode}")
    return max(0, round(raw * multiplier))


def rate_count(video: dict[str, str], peers: list[dict[str, str]], counts: Counter[str]) -> float:
    n_punches = sum(counts[row["video_key"]] for row in peers)
    n_seconds = sum(int(row["frame_count"]) / FPS for row in peers)
    return (n_punches / max(1e-6, n_seconds)) * (int(video["frame_count"]) / FPS)


def fight_group(video: dict[str, str]) -> str:
    return "|".join(
        [
            video["dataset_type"],
            video["data_root"],
            video["fight_index"],
            video["fight_folder"],
        ]
    )


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
