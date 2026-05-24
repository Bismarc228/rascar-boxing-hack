#!/usr/bin/env python3
"""Evaluate combined pose-context and ViT fixed-row keep/drop gates."""

from __future__ import annotations

import argparse
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

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_exchange_state_gate import index_candidates, row_features
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_vit_fixed_row_attribute_head import component_summary, fight_group, print_score
from tools.evaluate_vit_fixed_row_gate import apply_keep_mask, build_labels, labels_to_keep, print_label_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--feature-caches", type=Path, nargs="*")
    parser.add_argument("--feature-modes", default="pose,vit,pose_vit")
    parser.add_argument("--models", default="hgb,logreg")
    parser.add_argument("--ensemble-pair", default="")
    parser.add_argument("--ensemble-modes", default="drop_if_both_low,drop_if_either_low,mean,min,max")
    parser.add_argument("--label-positive", choices=["scorable", "matched"], default="scorable")
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument(
        "--thresholds",
        default="0.02,0.04,0.06,0.08,0.1,0.12,0.15,0.18,0.22,0.26,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-feature-mode", choices=["pose", "vit", "pose_vit"])
    parser.add_argument("--write-model", choices=["hgb", "logreg"])
    parser.add_argument("--write-threshold", type=float)
    parser.add_argument("--write-ensemble-mode")
    parser.add_argument("--write-left-threshold", type=float)
    parser.add_argument("--write-right-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    keys = sorted({row["video_key"] for row in pred_rows})
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    labels = build_labels(pred_rows, gt_rows)
    y = labels_to_keep(labels, args.label_positive)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])

    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, baseline, len(pred_rows), 0)
    oracle_mask = np.asarray([label != "fp" for label in labels], dtype=bool)
    oracle_rows = apply_keep_mask(pred_rows, oracle_mask)
    oracle_score = score_predictions(gt_rows, oracle_rows)
    print_score("oracle_drop_unmatched_fp", oracle_score, baseline, len(oracle_rows), video_wins(oracle_score, baseline))

    feature_modes = parse_list(args.feature_modes)
    models = parse_list(args.models)
    feature_blocks: dict[str, np.ndarray] = {}
    if any("pose" in mode for mode in feature_modes):
        feature_blocks["pose"] = build_pose_features(args, pred_rows, video_by_key, keys)
        print(f"pose_features_shape={feature_blocks['pose'].shape}", flush=True)
    if any("vit" in mode for mode in feature_modes):
        if not args.feature_caches:
            raise ValueError("feature modes with vit require --feature-caches")
        feature_blocks["vit"] = load_feature_caches(args.feature_caches)
        if len(feature_blocks["vit"]) != len(pred_rows):
            raise ValueError(f"feature rows {len(feature_blocks['vit'])} != prediction rows {len(pred_rows)}")
        print(f"vit_features_shape={feature_blocks['vit'].shape}", flush=True)

    thresholds = parse_floats(args.thresholds)
    probability_cache: dict[tuple[str, str], np.ndarray] = {}
    results: list[dict[str, Any]] = []
    for mode in feature_modes:
        x = select_features(mode, feature_blocks)
        for model_name in models:
            p_keep = fit_oof_probabilities(model_name, x, y, groups, mode)
            probability_cache[(mode, model_name)] = p_keep
            print(f"label_summary feature_mode={mode} model={model_name}")
            print_label_summary(labels, p_keep)
            for threshold in thresholds:
                add_result(
                    results,
                    gt_rows,
                    pred_rows,
                    baseline,
                    p_keep >= threshold,
                    feature_mode=mode,
                    model=model_name,
                    threshold=f"{threshold:g}",
                )

    if args.ensemble_pair:
        left, right = parse_ensemble_pair(args.ensemble_pair)
        left_probs = probability_cache[left]
        right_probs = probability_cache[right]
        for ensemble_mode in parse_list(args.ensemble_modes):
            for left_threshold in thresholds:
                for right_threshold in thresholds:
                    keep_mask = ensemble_keep_mask(
                        ensemble_mode,
                        left_probs,
                        right_probs,
                        left_threshold,
                        right_threshold,
                    )
                    add_result(
                        results,
                        gt_rows,
                        pred_rows,
                        baseline,
                        keep_mask,
                        feature_mode=f"ensemble_{ensemble_mode}",
                        model=args.ensemble_pair,
                        threshold=f"{left_threshold:g}|{right_threshold:g}",
                    )

    print("macro,delta,time,fighter,punch_type,effectiveness,hand,target,fp_penalty,wins,n_kept,n_dropped,feature_mode,model,threshold")
    for item in sorted(results, key=lambda row: row["macro"], reverse=True)[: args.top_k]:
        print(
            f"{item['macro']:.6f},{item['delta']:+.6f},{item['time']:.6f},{item['fighter']:.6f},"
            f"{item['punch_type']:.6f},{item['effectiveness']:.6f},{item['hand']:.6f},"
            f"{item['target']:.6f},{item['fp_penalty']:.6f},{item['wins']},"
            f"{item['n_kept']},{item['n_dropped']},{item['feature_mode']},{item['model']},{item['threshold']}"
        )

    if args.write_oof_rows:
        if args.write_ensemble_mode:
            if not args.ensemble_pair:
                raise ValueError("--write-ensemble-mode requires --ensemble-pair")
            if args.write_left_threshold is None or args.write_right_threshold is None:
                raise ValueError("--write-ensemble-mode requires --write-left-threshold and --write-right-threshold")
            left, right = parse_ensemble_pair(args.ensemble_pair)
            keep_mask = ensemble_keep_mask(
                args.write_ensemble_mode,
                probability_cache[left],
                probability_cache[right],
                args.write_left_threshold,
                args.write_right_threshold,
            )
            rows = apply_keep_mask(pred_rows, keep_mask)
            write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
            print(
                f"wrote_oof_rows={args.write_oof_rows} ensemble_mode={args.write_ensemble_mode} "
                f"pair={args.ensemble_pair} left_threshold={args.write_left_threshold} "
                f"right_threshold={args.write_right_threshold} n_rows={len(rows)}"
            )
            return 0
        best = max(results, key=lambda row: row["macro"])
        mode = args.write_feature_mode or str(best["feature_mode"])
        model_name = args.write_model or str(best["model"])
        threshold = args.write_threshold if args.write_threshold is not None else float(best["threshold"])
        p_keep = probability_cache[(mode, model_name)]
        rows = apply_keep_mask(pred_rows, p_keep >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} feature_mode={mode} "
            f"model={model_name} threshold={threshold} n_rows={len(rows)}"
        )
    return 0


def add_result(
    results: list[dict[str, Any]],
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    baseline: dict[str, Any],
    keep_mask: np.ndarray,
    *,
    feature_mode: str,
    model: str,
    threshold: str,
) -> None:
    rows = apply_keep_mask(pred_rows, keep_mask)
    score = score_predictions(gt_rows, rows)
    summary = component_summary(score)
    results.append(
        {
            "macro": score["macro_score"],
            "delta": score["macro_score"] - baseline["macro_score"],
            "time": summary["time"],
            "fighter": summary["fighter"],
            "punch_type": summary["punch_type"],
            "effectiveness": summary["effectiveness"],
            "hand": summary["hand"],
            "target": summary["target"],
            "fp_penalty": summary["fp_penalty"],
            "wins": video_wins(score, baseline),
            "n_kept": len(rows),
            "n_dropped": int((~keep_mask).sum()),
            "feature_mode": feature_mode,
            "model": model,
            "threshold": threshold,
        }
    )


def build_pose_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    keys: list[str],
) -> np.ndarray:
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    candidates_by_key = {
        key: apply_temporal_context(score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config), config)
        for key in keys
    }
    indexed_by_key = {key: index_candidates(candidates) for key, candidates in candidates_by_key.items()}
    windows = parse_ints(args.feature_windows)
    return np.stack(
        [
            row_features(
                row,
                video_by_key[row["video_key"]],
                indexed_by_key[row["video_key"]],
                args.match_window,
                windows,
            )
            for row in pred_rows
        ]
    ).astype(np.float32)


def select_features(mode: str, feature_blocks: dict[str, np.ndarray]) -> np.ndarray:
    if mode == "pose":
        return feature_blocks["pose"]
    if mode == "vit":
        return feature_blocks["vit"]
    if mode == "pose_vit":
        return np.concatenate([feature_blocks["pose"], feature_blocks["vit"]], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature mode: {mode}")


def parse_ensemble_pair(text: str) -> tuple[tuple[str, str], tuple[str, str]]:
    parts = parse_list(text)
    if len(parts) != 2:
        raise ValueError("--ensemble-pair must look like pose:hgb,vit:logreg")
    output = []
    for part in parts:
        mode, sep, model = part.partition(":")
        if sep != ":" or not mode or not model:
            raise ValueError("--ensemble-pair must look like pose:hgb,vit:logreg")
        output.append((mode, model))
    return output[0], output[1]


def ensemble_keep_mask(
    mode: str,
    left_probs: np.ndarray,
    right_probs: np.ndarray,
    left_threshold: float,
    right_threshold: float,
) -> np.ndarray:
    if mode == "drop_if_both_low":
        return (left_probs >= left_threshold) | (right_probs >= right_threshold)
    if mode == "drop_if_either_low":
        return (left_probs >= left_threshold) & (right_probs >= right_threshold)
    if mode == "mean":
        return (left_probs + right_probs) / 2.0 >= (left_threshold + right_threshold) / 2.0
    if mode == "min":
        return np.minimum(left_probs, right_probs) >= min(left_threshold, right_threshold)
    if mode == "max":
        return np.maximum(left_probs, right_probs) >= max(left_threshold, right_threshold)
    raise ValueError(f"unknown ensemble mode: {mode}")


def load_feature_caches(paths: list[Path]) -> np.ndarray:
    arrays = []
    for path in paths:
        data = np.load(path)
        features = data["features"].astype(np.float32)
        if features.ndim != 3:
            raise ValueError(f"expected [N,T,D] in {path}, got {features.shape}")
        mean = features.mean(axis=1)
        std = features.std(axis=1)
        center = features[:, features.shape[1] // 2]
        arrays.append(np.concatenate([center, mean, std], axis=1).astype(np.float32))
    rows = {array.shape[0] for array in arrays}
    if len(rows) != 1:
        raise ValueError(f"feature caches have different row counts: {sorted(rows)}")
    return np.concatenate(arrays, axis=1).astype(np.float32)


def fit_oof_probabilities(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_mode: str,
) -> np.ndarray:
    p = np.zeros(len(y), dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = ~valid
        if len(np.unique(y[train])) < 2:
            p[valid] = float(y[train].mean()) if train.any() else 1.0
            continue
        model = fit_model(model_name, x[train], y[train])
        p[valid] = model.predict_proba(x[valid])[:, 1].astype(np.float32)
        print(
            f"fold={group} feature_mode={feature_mode} model={model_name} "
            f"train={int(train.sum())} valid={int(valid.sum())} pos_rate={float(y[train].mean()):.4f}",
            flush=True,
        )
    return p


def fit_model(model_name: str, x: np.ndarray, y: np.ndarray):
    if model_name == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1200, C=0.35),
        )
    elif model_name == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.5,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(x, y)
    return model


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
