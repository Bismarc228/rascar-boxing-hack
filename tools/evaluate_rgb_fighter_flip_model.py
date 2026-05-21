#!/usr/bin/env python3
"""Evaluate fixed-row fighter flips from cached row-aligned RGB clip features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from tools.evaluate_exchange_state_gate import fight_group
from tools.evaluate_fighter_rival_flip import score_summary
from tools.evaluate_pose_selection_variants import video_wins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--label-window", type=int, default=int(FPS * 0.5))
    parser.add_argument("--models", default="logreg,hgb")
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90,0.95,0.98")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--logreg-c", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-model", default="")
    parser.add_argument("--write-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    if not pred_rows:
        print("No clear prediction rows found")
        return 2
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    missing = sorted(set(keys) - set(video_by_key))
    if missing:
        raise RuntimeError(f"Prediction keys are not train videos: {missing}")
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, 0, len(pred_rows))

    features = load_features(args.feature_cache, len(pred_rows))
    x = build_features(pred_rows, video_by_key, features)
    labels, matched = fighter_flip_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    print(
        f"features shape={features.shape} x_shape={x.shape} "
        f"matched_labels={int(matched.sum())} flip_labels={int(labels.sum())}",
        flush=True,
    )

    probabilities_by_model: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    for model_name in parse_list(args.models):
        probabilities = fit_oof_probabilities(args, model_name, x, labels, groups)
        probabilities_by_model[model_name] = probabilities
        print_probability_summary(model_name, labels, probabilities)
        for threshold in parse_floats(args.thresholds):
            rows, changed = apply_fighter_flips(pred_rows, probabilities >= threshold)
            if changed == 0:
                continue
            score = score_predictions(gt_rows, rows)
            summary = score_summary(score)
            results.append(
                {
                    "score": score["macro_score"],
                    "delta": score["macro_score"] - baseline["macro_score"],
                    "fighter": summary["fighter"],
                    "time": summary["time"],
                    "hand": summary["hand"],
                    "target": summary["target"],
                    "fp": summary["fp"],
                    "wins": video_wins(score, baseline),
                    "changed": changed,
                    "model": model_name,
                    "threshold": threshold,
                }
            )

    print("score,delta,fighter,time,hand,target,fp,wins,n_changed,model,threshold")
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print(
            f"{item['score']:.6f},{item['delta']:.6f},{item['fighter']:.6f},"
            f"{item['time']:.6f},{item['hand']:.6f},{item['target']:.6f},"
            f"{item['fp']:.6f},{item['wins']},{item['changed']},"
            f"{item['model']},{item['threshold']}",
            flush=True,
        )

    if args.write_oof_rows:
        if args.write_model and args.write_threshold is not None:
            model_name = args.write_model
            threshold = args.write_threshold
        elif results:
            best = max(results, key=lambda row: row["score"])
            model_name = str(best["model"])
            threshold = float(best["threshold"])
        else:
            print("No positive flip result; not writing OOF rows")
            return 0
        rows, changed = apply_fighter_flips(pred_rows, probabilities_by_model[model_name] >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} model={model_name} "
            f"threshold={threshold} changed={changed}",
            flush=True,
        )
    return 0


def load_features(path: Path, expected_rows: int) -> np.ndarray:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    if features.ndim != 3:
        raise ValueError(f"expected feature tensor [N,T,D], got {features.shape}")
    if len(features) != expected_rows:
        raise ValueError(f"feature rows {len(features)} != prediction rows {expected_rows}")
    return features


def build_features(
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    features: np.ndarray,
) -> np.ndarray:
    vectors = []
    for row, clip in zip(rows, features):
        rgb = np.concatenate(
            [
                clip.mean(axis=0),
                clip.std(axis=0),
                clip[len(clip) // 2],
                clip[-1] - clip[0],
            ]
        ).astype(np.float32)
        vectors.append(np.concatenate([rgb, row_context_features(row, video_by_key[row["video_key"]])]))
    return np.stack(vectors).astype(np.float32)


def row_context_features(row: dict[str, str], video: dict[str, str]) -> np.ndarray:
    values = [
        1.0 if row["fighter"] == "red" else 0.0,
        1.0 if row["hand"] == "left" else 0.0,
        1.0 if row["target"] == "head" else 0.0,
        1.0 if row["punch_type"] == "jab" else 0.0,
        1.0 if row["punch_type"] == "cross" else 0.0,
        1.0 if row["punch_type"] == "hook" else 0.0,
        1.0 if row["punch_type"] == "uppercut" else 0.0,
        1.0 if row["effectiveness"] == "landed" else 0.0,
        1.0 if row["effectiveness"] == "blocked" else 0.0,
        1.0 if row["effectiveness"] in {"miss", "missed"} else 0.0,
        as_int(row["frame"], "frame") / max(1.0, float(video["frame_count"])),
    ]
    return np.asarray(values, dtype=np.float32)


def fighter_flip_labels(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    label_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(pred_rows), dtype=np.int8)
    matched = np.zeros(len(pred_rows), dtype=bool)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    global_indices_by_key: dict[str, list[int]] = {}
    for index, row in enumerate(pred_rows):
        global_indices_by_key.setdefault(row["video_key"], []).append(index)
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            pred = rows[match.pred_index]
            gt = gt_by_key[key][match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) > label_window:
                continue
            global_index = global_indices_by_key[key][match.pred_index]
            matched[global_index] = True
            if pred["fighter"] != gt["fighter"]:
                labels[global_index] = 1
    return labels, matched


def fit_oof_probabilities(
    args: argparse.Namespace,
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    probabilities = np.zeros(len(y), dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = groups != group
        if train.sum() < 20 or valid.sum() == 0 or y[train].sum() == 0:
            print(
                f"fold={group} model={model_name} train={int(train.sum())} "
                f"valid={int(valid.sum())} positives={int(y[train].sum())} skipped",
                flush=True,
            )
            continue
        model = make_model(args, model_name, int(train.sum()), x.shape[1])
        model.fit(x[train], y[train])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x[valid])
            classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
            class_index = classes.index(1) if 1 in classes else None
            if class_index is not None:
                probabilities[valid] = proba[:, class_index]
        print(
            f"fold={group} model={model_name} train={int(train.sum())} "
            f"valid={int(valid.sum())} positives={int(y[train].sum())}",
            flush=True,
        )
    return probabilities


def make_model(args: argparse.Namespace, model_name: str, n_train: int, n_features: int):
    n_components = min(args.pca_components, n_train - 1, n_features)
    if n_components < 1:
        raise RuntimeError(f"Not enough rows for PCA: n_train={n_train}")
    if model_name == "logreg":
        classifier = LogisticRegression(
            C=args.logreg_c,
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "hgb":
        classifier = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.04,
            l2_regularization=0.02,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=42), classifier)


def apply_fighter_flips(rows: list[dict[str, str]], mask: np.ndarray) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for row, should_flip in zip(rows, mask):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if should_flip:
            item["fighter"] = opposite_fighter(row["fighter"])
        if item["fighter"] != row["fighter"]:
            changed += 1
        output.append(item)
    return output, changed


def opposite_fighter(value: str) -> str:
    if value == "red":
        return "blue"
    if value == "blue":
        return "red"
    raise ValueError(f"unknown fighter {value!r}")


def print_score(label: str, score: dict[str, Any], changed: int, n_rows: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},fighter={summary['fighter']:.6f},"
        f"time={summary['time']:.6f},hand={summary['hand']:.6f},"
        f"target={summary['target']:.6f},fp={summary['fp']:.6f},"
        f"changed={changed},n={n_rows}",
        flush=True,
    )


def print_probability_summary(model_name: str, labels: np.ndarray, probabilities: np.ndarray) -> None:
    pos = labels == 1
    neg = labels == 0
    pos_mean = float(probabilities[pos].mean()) if pos.any() else 0.0
    neg_mean = float(probabilities[neg].mean()) if neg.any() else 0.0
    print(
        f"model={model_name} positives={int(pos.sum())} negatives={int(neg.sum())} "
        f"pos_mean_p={pos_mean:.4f} neg_mean_p={neg_mean:.4f} "
        f"max_p={float(probabilities.max()):.4f}",
        flush=True,
    )


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
