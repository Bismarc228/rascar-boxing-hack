#!/usr/bin/env python3
"""Evaluate an OOF exchange-state keep/drop gate for fixed prediction rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    score_pose_tracks,
)
from tools.evaluate_fighter_rival_flip import best_candidate, index_candidates
from tools.evaluate_pose_selection_variants import video_wins


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument(
        "--thresholds",
        default="0.02,0.04,0.06,0.08,0.1,0.12,0.15,0.18,0.22,0.26,0.3,0.35,0.4,0.5,0.6",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, 0, len(pred_rows))

    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    candidates_by_key = {
        key: apply_temporal_context(
            score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config),
            config,
        )
        for key in keys
    }
    indexed_by_key = {key: index_candidates(candidates) for key, candidates in candidates_by_key.items()}
    windows = parse_ints(args.feature_windows)
    x = np.stack(
        [
            row_features(row, video_by_key[row["video_key"]], indexed_by_key[row["video_key"]], args.match_window, windows)
            for row in pred_rows
        ]
    )
    y, labels = build_labels(pred_rows, gt_rows)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    p_keep = fit_oof_probabilities(args.model, x, y, groups)

    oracle_fp_rows = apply_keep_mask(pred_rows, np.asarray([label != "fp" for label in labels], dtype=bool))
    oracle_fp_score = score_predictions(gt_rows, oracle_fp_rows)
    print_score("oracle_drop_unmatched_fp", oracle_fp_score, video_wins(oracle_fp_score, baseline), len(oracle_fp_rows))
    print_label_summary(labels, p_keep)

    results = []
    for threshold in parse_floats(args.thresholds):
        keep_mask = p_keep >= threshold
        rows = apply_keep_mask(pred_rows, keep_mask)
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        dropped = int((~keep_mask).sum())
        results.append(
            {
                "score": score["macro_score"],
                "delta": score["macro_score"] - baseline["macro_score"],
                "fighter": summary["fighter"],
                "time": summary["time"],
                "fp": summary["fp"],
                "wins": video_wins(score, baseline),
                "n_kept": len(rows),
                "n_dropped": dropped,
                "threshold": threshold,
            }
        )

    print("score,delta,fighter,time,fp,wins,n_kept,n_dropped,threshold")
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print(
            f"{item['score']:.6f},{item['delta']:.6f},{item['fighter']:.6f},"
            f"{item['time']:.6f},{item['fp']:.6f},{item['wins']},"
            f"{item['n_kept']},{item['n_dropped']},{item['threshold']}"
        )
    if args.write_oof_rows:
        threshold = args.write_threshold
        if threshold is None:
            threshold = max(results, key=lambda row: row["score"])["threshold"]
        rows = apply_keep_mask(pred_rows, p_keep >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} threshold={threshold} n_rows={len(rows)}")
    return 0


def row_features(
    row: dict[str, str],
    video: dict[str, str],
    candidates_by_frame: dict[int, list[PunchCandidate]],
    match_window: int,
    windows: list[int],
) -> np.ndarray:
    frame = as_int(row["frame"], "frame")
    self_candidate = best_candidate(
        candidates_by_frame,
        frame,
        match_window,
        lambda candidate: candidate.fighter == row["fighter"]
        and candidate.hand == row["hand"]
        and candidate.target == row["target"],
    )
    if self_candidate is None:
        self_candidate = best_candidate(
            candidates_by_frame,
            frame,
            match_window,
            lambda candidate: candidate.fighter == row["fighter"] and candidate.hand == row["hand"],
        )
    if self_candidate is None:
        self_candidate = best_candidate(
            candidates_by_frame,
            frame,
            match_window,
            lambda candidate: candidate.fighter == row["fighter"],
        )
    rival_candidate = best_candidate(
        candidates_by_frame,
        frame,
        match_window,
        lambda candidate: candidate.fighter != row["fighter"],
    )
    self_score = float(self_candidate.score) if self_candidate else 0.0
    rival_score = float(rival_candidate.score) if rival_candidate else 0.0
    values = [
        float(np.log1p(max(0.0, self_score))),
        self_score,
        rival_score,
        rival_score / max(1e-6, self_score),
        float(frame / max(1.0, float(video["frame_count"]))),
        1.0 if row["fighter"] == "red" else 0.0,
        1.0 if row["hand"] == "left" else 0.0,
        1.0 if row["target"] == "head" else 0.0,
    ]
    for root in ["бокс", "Турнир Бокс", "Турнир Бокс 2"]:
        values.append(float(video["data_root"] == root))
    for name in BASE_FEATURES:
        raw = float(self_candidate.features.get(name, 0.0)) if self_candidate else 0.0
        if name == "forward_norm":
            raw /= 1000.0
        values.append(raw)

    for window in windows:
        stats = window_stats(candidates_by_frame, frame, window, row)
        values.extend(stats)
    return np.asarray(values, dtype=np.float32)


def window_stats(
    candidates_by_frame: dict[int, list[PunchCandidate]],
    frame: int,
    window: int,
    row: dict[str, str],
) -> list[float]:
    all_scores = []
    same_fighter_scores = []
    same_group_scores = []
    rival_scores = []
    for current in range(frame - window, frame + window + 1):
        for candidate in candidates_by_frame.get(current, []):
            all_scores.append(candidate.score)
            if candidate.fighter == row["fighter"]:
                same_fighter_scores.append(candidate.score)
                if candidate.hand == row["hand"]:
                    same_group_scores.append(candidate.score)
            else:
                rival_scores.append(candidate.score)
    all_sum = float(sum(all_scores))
    same_sum = float(sum(same_fighter_scores))
    same_group_sum = float(sum(same_group_scores))
    rival_sum = float(sum(rival_scores))
    return [
        all_sum,
        float(len(all_scores)),
        max(all_scores) if all_scores else 0.0,
        same_sum,
        float(len(same_fighter_scores)),
        same_group_sum,
        float(len(same_group_scores)),
        rival_sum,
        float(len(rival_scores)),
        max(rival_scores) if rival_scores else 0.0,
        same_sum / max(1e-6, all_sum),
        rival_sum / max(1e-6, all_sum),
    ]


def build_labels(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[str]]:
    gt_by_key = group_by(gt_rows, "video_key")
    pred_by_key = group_by(pred_rows, "video_key")
    labels = ["fp"] * len(pred_rows)
    global_indices_by_key: dict[str, list[int]] = defaultdict(list)
    for global_index, row in enumerate(pred_rows):
        global_indices_by_key[row["video_key"]].append(global_index)
    for key, rows in pred_by_key.items():
        gt_video = gt_by_key.get(key, [])
        for match in match_events(gt_video, rows):
            pred = rows[match.pred_index]
            gt = gt_video[match.gt_index]
            frame_error = abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame"))
            global_index = global_indices_by_key[key][match.pred_index]
            labels[global_index] = "tp_scorable" if frame_error / FPS / 0.5 < 1.0 else "tp_time_only"
    y = np.asarray([1 if label == "tp_scorable" else 0 for label in labels], dtype=np.int8)
    return y, labels


def fit_oof_probabilities(model_name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    p = np.zeros(len(y), dtype=np.float32)
    for group in sorted(set(groups)):
        valid = groups == group
        train = ~valid
        if len(np.unique(y[train])) < 2:
            p[valid] = float(y[train].mean()) if train.any() else 1.0
            continue
        model = fit_model(model_name, x[train], y[train])
        p[valid] = model.predict_proba(x[valid])[:, 1].astype(np.float32)
        print(
            f"fold={group} train={int(train.sum())} valid={int(valid.sum())} "
            f"pos_rate={float(y[train].mean()):.4f}",
            flush=True,
        )
    return p


def fit_model(model_name: str, x: np.ndarray, y: np.ndarray):
    if model_name == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5),
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.5,
            random_state=42,
        )
    model.fit(x, y)
    return model


def apply_keep_mask(rows: list[dict[str, str]], keep_mask: np.ndarray) -> list[dict[str, str]]:
    output = []
    for row, keep in zip(rows, keep_mask):
        if not keep:
            continue
        output.append({col: row.get(col, "") for col in SUBMISSION_COLUMNS})
    return output


def print_label_summary(labels: list[str], p_keep: np.ndarray) -> None:
    counts = Counter(labels)
    print(
        "labels="
        + ",".join(f"{label}:{counts[label]}" for label in ["tp_scorable", "tp_time_only", "fp"])
    )
    for label in ["tp_scorable", "tp_time_only", "fp"]:
        values = p_keep[np.asarray([item == label for item in labels], dtype=bool)]
        if len(values) == 0:
            continue
        print(
            f"p_keep_{label}=n:{len(values)},mean:{float(values.mean()):.4f},"
            f"p10:{float(np.quantile(values, 0.1)):.4f},"
            f"p50:{float(np.quantile(values, 0.5)):.4f},"
            f"p90:{float(np.quantile(values, 0.9)):.4f}"
        )


def print_score(label: str, score: dict[str, object], wins: int, n_rows: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},fighter={summary['fighter']:.6f},"
        f"time={summary['time']:.6f},fp={summary['fp']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fp": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def fight_group(video: dict[str, str]) -> str:
    return "|".join([video["dataset_type"], video["data_root"], video["fight_index"], video["fight_folder"]])


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
