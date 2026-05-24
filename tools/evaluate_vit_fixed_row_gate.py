#!/usr/bin/env python3
"""Evaluate fixed-row keep/drop gates on cached ViT/VideoMAE features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

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
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_vit_fixed_row_attribute_head import component_summary, fight_group, print_score
from tools.evaluate_vit_neural_attribute_head import load_feature_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-caches", type=Path, nargs="+", required=True)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--label-positive", choices=["scorable", "matched"], default="scorable")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    keys = {row["video_key"] for row in pred_rows}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in keys
    ]
    x = load_feature_caches(args.feature_caches)
    if len(x) != len(pred_rows):
        raise ValueError(f"feature rows {len(x)} != prediction rows {len(pred_rows)}")
    labels = build_labels(pred_rows, gt_rows)
    y = labels_to_keep(labels, args.label_positive)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])

    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, baseline, len(pred_rows), 0)
    oracle_mask = np.asarray([label != "fp" for label in labels], dtype=bool)
    oracle_rows = apply_keep_mask(pred_rows, oracle_mask)
    oracle_score = score_predictions(gt_rows, oracle_rows)
    print_score("oracle_drop_unmatched_fp", oracle_score, baseline, len(oracle_rows), video_wins(oracle_score, baseline))
    p_keep = fit_oof_probabilities(args.model, x, y, groups)
    print_label_summary(labels, p_keep)

    results = []
    for threshold in parse_floats(args.thresholds):
        keep_mask = p_keep >= threshold
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
                "threshold": threshold,
            }
        )

    print("macro,delta,time,fighter,punch_type,effectiveness,hand,target,fp_penalty,wins,n_kept,n_dropped,threshold")
    for item in sorted(results, key=lambda row: row["macro"], reverse=True)[: args.top_k]:
        print(
            f"{item['macro']:.6f},{item['delta']:+.6f},{item['time']:.6f},{item['fighter']:.6f},"
            f"{item['punch_type']:.6f},{item['effectiveness']:.6f},{item['hand']:.6f},"
            f"{item['target']:.6f},{item['fp_penalty']:.6f},{item['wins']},"
            f"{item['n_kept']},{item['n_dropped']},{item['threshold']}"
        )
    if args.write_oof_rows:
        threshold = args.write_threshold
        if threshold is None:
            threshold = max(results, key=lambda row: row["macro"])["threshold"]
        rows = apply_keep_mask(pred_rows, p_keep >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} threshold={threshold} n_rows={len(rows)}")
    return 0


def build_labels(pred_rows: list[dict[str, str]], gt_rows: list[dict[str, str]]) -> list[str]:
    gt_by_key = group_by(gt_rows, "video_key")
    pred_by_key = group_by(pred_rows, "video_key")
    labels = ["fp"] * len(pred_rows)
    global_indices_by_key: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        global_indices_by_key[row["video_key"]].append(index)
    for key, rows in pred_by_key.items():
        gt_video = gt_by_key.get(key, [])
        for match in match_events(gt_video, rows):
            pred = rows[match.pred_index]
            gt = gt_video[match.gt_index]
            frame_error = abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame"))
            global_index = global_indices_by_key[key][match.pred_index]
            labels[global_index] = "tp_scorable" if frame_error / FPS / 0.5 < 1.0 else "tp_time_only"
    return labels


def labels_to_keep(labels: list[str], positive: str) -> np.ndarray:
    if positive == "matched":
        return np.asarray([label != "fp" for label in labels], dtype=np.int8)
    return np.asarray([label == "tp_scorable" for label in labels], dtype=np.int8)


def fit_oof_probabilities(model_name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
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
            f"fold={group} train={int(train.sum())} valid={int(valid.sum())} "
            f"pos_rate={float(y[train].mean()):.4f}",
            flush=True,
        )
    return p


def fit_model(model_name: str, x: np.ndarray, y: np.ndarray):
    if model_name == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1200, C=0.35),
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=0.8,
            random_state=42,
        )
    model.fit(x, y)
    return model


def apply_keep_mask(rows: list[dict[str, str]], keep_mask: np.ndarray) -> list[dict[str, str]]:
    return [
        {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for row, keep in zip(rows, keep_mask)
        if keep
    ]


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


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
