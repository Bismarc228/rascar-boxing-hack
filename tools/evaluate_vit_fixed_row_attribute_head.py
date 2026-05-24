#!/usr/bin/env python3
"""Evaluate fixed-row attribute heads on extracted ViT feature caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from tools.evaluate_pose_selection_variants import video_wins


ATTR_COLUMNS = ["punch_type", "effectiveness", "hand", "target", "fighter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=96)
    parser.add_argument("--logreg-c", type=float, default=0.35)
    parser.add_argument("--heads", default="effectiveness,punch_type")
    parser.add_argument("--margins", default="0.0,0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--combos", default="")
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-head", choices=ATTR_COLUMNS)
    parser.add_argument("--write-margin", type=float)
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
    x = load_features(args.feature_cache)
    if len(x) != len(pred_rows):
        raise ValueError(f"feature rows {len(x)} != prediction rows {len(pred_rows)}")
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])

    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, baseline, len(pred_rows), 0)
    predictions, probabilities = oof_predictions(args, x, labels, groups, pred_rows)
    print_head_diagnostics(labels, pred_rows, predictions, parse_list(args.heads))

    rows_by_variant = {}
    print("variant,macro,delta,time,fighter,punch_type,effectiveness,hand,target,fp_penalty,wins,changed")
    for head in [item for item in args.heads.split(",") if item]:
        for margin in parse_floats(args.margins):
            variant = f"{head}_margin_{margin:g}"
            rows, changed = apply_margin(pred_rows, predictions, probabilities, head, margin)
            rows_by_variant[(head, margin)] = rows
            score = score_predictions(gt_rows, rows)
            summary = component_summary(score)
            print(
                f"{variant},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:+.6f},"
                f"{summary['time']:.6f},{summary['fighter']:.6f},{summary['punch_type']:.6f},"
                f"{summary['effectiveness']:.6f},{summary['hand']:.6f},{summary['target']:.6f},"
                f"{summary['fp_penalty']:.6f},{video_wins(score, baseline)},{changed}",
                flush=True,
            )
    for combo in parse_combos(args.combos):
        variant = "combo_" + "_".join(f"{head}{margin:g}" for head, margin in combo)
        rows, changed = apply_combo(pred_rows, predictions, probabilities, combo)
        score = score_predictions(gt_rows, rows)
        summary = component_summary(score)
        print(
            f"{variant},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:+.6f},"
            f"{summary['time']:.6f},{summary['fighter']:.6f},{summary['punch_type']:.6f},"
            f"{summary['effectiveness']:.6f},{summary['hand']:.6f},{summary['target']:.6f},"
            f"{summary['fp_penalty']:.6f},{video_wins(score, baseline)},{changed}",
            flush=True,
        )
    print_label_counts(labels)
    if args.write_oof_rows:
        if args.write_head is None or args.write_margin is None:
            raise ValueError("--write-oof-rows requires --write-head and --write-margin")
        rows = rows_by_variant[(args.write_head, args.write_margin)]
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} head={args.write_head} margin={args.write_margin}")
    return 0


def load_features(path: Path) -> np.ndarray:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    if features.ndim != 3:
        raise ValueError(f"expected [N,T,D], got {features.shape}")
    mean = features.mean(axis=1)
    std = features.std(axis=1)
    center = features[:, features.shape[1] // 2]
    return np.concatenate([center, mean, std], axis=1).astype(np.float32)


def matched_attribute_labels(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    label_window: int,
) -> list[dict[str, str] | None]:
    labels: list[dict[str, str] | None] = [None] * len(pred_rows)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    global_indices: dict[str, list[int]] = {}
    for index, row in enumerate(pred_rows):
        global_indices.setdefault(row["video_key"], []).append(index)
    for key, rows in pred_by_key.items():
        gt_video = gt_by_key.get(key, [])
        for match in match_events(gt_video, rows):
            pred = rows[match.pred_index]
            gt = gt_video[match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) > label_window:
                continue
            labels[global_indices[key][match.pred_index]] = gt
    return labels


def oof_predictions(
    args: argparse.Namespace,
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    groups: np.ndarray,
    pred_rows: list[dict[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, float]]]]:
    output = {head: np.asarray([row[head] for row in pred_rows], dtype=object) for head in ATTR_COLUMNS}
    probabilities = {head: [{row[head]: 1.0} for row in pred_rows] for head in ATTR_COLUMNS}
    trainable = np.asarray([label is not None for label in labels], dtype=bool)
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = (groups != group) & trainable
        valid_predict = valid
        if train.sum() < 20:
            print(f"fold={group} train={int(train.sum())} skipped", flush=True)
            continue
        for head in ATTR_COLUMNS:
            encoder = LabelEncoder()
            values = [labels[index][head] for index in np.where(train)[0]]  # type: ignore[index]
            y = encoder.fit_transform(values)
            if len(encoder.classes_) == 1:
                continue
            n_components = min(args.pca_components, train.sum() - 1, x.shape[1])
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=42),
                LogisticRegression(
                    C=args.logreg_c,
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=42,
                ),
            )
            model.fit(x[train], y)
            pred_encoded = model.predict(x[valid_predict])
            output[head][valid_predict] = encoder.inverse_transform(pred_encoded)
            proba = model.predict_proba(x[valid_predict])
            class_values = [str(value) for value in encoder.classes_]
            for index, row_probs in zip(np.where(valid_predict)[0], proba):
                probabilities[head][index] = {
                    class_value: float(prob)
                    for class_value, prob in zip(class_values, row_probs)
                }
        print(f"fold={group} train={int(train.sum())} valid={int(valid_predict.sum())}", flush=True)
    return output, probabilities


def apply_margin(
    rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, list[dict[str, float]]],
    head: str,
    margin: float,
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        pred_value = str(predictions[head][index])
        current_value = row[head]
        probs = probabilities[head][index]
        pred_prob = float(probs.get(pred_value, 0.0))
        current_prob = float(probs.get(current_value, 0.0))
        if pred_value != current_value and pred_prob - current_prob >= margin:
            item[head] = pred_value
            changed += 1
        output.append(item)
    return output, changed


def apply_combo(
    rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, list[dict[str, float]]],
    combo: list[tuple[str, float]],
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for head, margin in combo:
            pred_value = str(predictions[head][index])
            current_value = row[head]
            probs = probabilities[head][index]
            pred_prob = float(probs.get(pred_value, 0.0))
            current_prob = float(probs.get(current_value, 0.0))
            if pred_value != current_value and pred_prob - current_prob >= margin:
                item[head] = pred_value
                changed += 1
        output.append(item)
    return output, changed


def component_summary(score: dict[str, Any]) -> dict[str, float]:
    values = list(score["by_video"].values())
    return {
        "time": float(np.mean([item["score_time"] for item in values])),
        "fighter": float(np.mean([item["score_fighter"] for item in values])),
        "punch_type": float(np.mean([item["score_punch_type"] for item in values])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in values])),
        "hand": float(np.mean([item["score_hand"] for item in values])),
        "target": float(np.mean([item["score_target"] for item in values])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in values])),
    }


def print_score(label: str, score: dict[str, Any], baseline: dict[str, Any], n_rows: int, wins: int) -> None:
    summary = component_summary(score)
    print(
        f"{label}: macro={score['macro_score']:.6f},delta={score['macro_score'] - baseline['macro_score']:+.6f},"
        f"time={summary['time']:.6f},fighter={summary['fighter']:.6f},"
        f"punch_type={summary['punch_type']:.6f},effectiveness={summary['effectiveness']:.6f},"
        f"hand={summary['hand']:.6f},target={summary['target']:.6f},"
        f"fp_penalty={summary['fp_penalty']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def print_label_counts(labels: list[dict[str, str] | None]) -> None:
    usable = [label for label in labels if label is not None]
    print(f"labeled_rows={len(usable)}")
    for head in ATTR_COLUMNS:
        counts: dict[str, int] = {}
        for label in usable:
            counts[label[head]] = counts.get(label[head], 0) + 1
        print("labels_" + head + "=" + ",".join(f"{key}:{value}" for key, value in sorted(counts.items())))


def print_head_diagnostics(
    labels: list[dict[str, str] | None],
    pred_rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    heads: list[str],
) -> None:
    indices = [index for index, label in enumerate(labels) if label is not None]
    print("diagnostic,head,source,accuracy,balanced_acc,macro_f1,weighted_f1")
    for head in heads:
        y_true = [labels[index][head] for index in indices]  # type: ignore[index]
        y_base = [pred_rows[index][head] for index in indices]
        y_model = [str(predictions[head][index]) for index in indices]
        print_diagnostic_row(head, "baseline", y_true, y_base)
        print_diagnostic_row(head, "raw_model", y_true, y_model)
        print_per_class_rows(head, "raw_model", y_true, y_model)


def print_diagnostic_row(head: str, source: str, y_true: list[str], y_pred: list[str]) -> None:
    print(
        f"diagnostic,{head},{source},{accuracy_score(y_true, y_pred):.6f},"
        f"{balanced_accuracy_score(y_true, y_pred):.6f},"
        f"{macro_f1(y_true, y_pred):.6f},{weighted_f1(y_true, y_pred):.6f}",
        flush=True,
    )


def print_per_class_rows(head: str, source: str, y_true: list[str], y_pred: list[str]) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    for label, p_value, r_value, f1_value, support_value in zip(labels, precision, recall, f1, support):
        print(
            f"diagnostic_class,{head},{source},{label},precision={p_value:.6f},"
            f"recall={r_value:.6f},f1={f1_value:.6f},support={int(support_value)}",
            flush=True,
        )


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2])


def weighted_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2])


def fight_group(video: dict[str, str]) -> str:
    return "|".join([video["dataset_type"], video["data_root"], video["fight_index"], video["fight_folder"]])


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_combos(text: str) -> list[list[tuple[str, float]]]:
    combos = []
    for combo_text in text.split(";"):
        combo_text = combo_text.strip()
        if not combo_text:
            continue
        combo = []
        for item in combo_text.split(","):
            head, margin = item.split(":", 1)
            head = head.strip()
            if head not in ATTR_COLUMNS:
                raise ValueError(f"unknown combo head {head}")
            combo.append((head, float(margin)))
        combos.append(combo)
    return combos


if __name__ == "__main__":
    raise SystemExit(main())
