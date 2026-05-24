#!/usr/bin/env python3
"""Evaluate frozen VideoMAE fixed-row features for attribute replacement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_rgb_fixed_row_attribute_model import (
    ATTR_COLUMNS,
    apply_attribute_margin,
    apply_predictions,
    attr_summary,
    matched_attribute_labels,
    oof_attribute_predictions,
    parse_floats,
    print_label_counts,
    print_score,
    row_context_features,
)
from tools.evaluate_rgb_timing_offset import fight_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path)
    parser.add_argument("--pooling", choices=["meanstd", "mean", "flatten"], default="meanstd")
    parser.add_argument("--no-row-context", action="store_true")
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--head-type", choices=["logreg", "torch_linear", "torch_mlp"], default="logreg")
    parser.add_argument("--train-columns", default="punch_type,effectiveness,target")
    parser.add_argument("--margin-columns", default="punch_type,effectiveness,target")
    parser.add_argument("--effectiveness-margins", default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--pca-components", type=int, default=160)
    parser.add_argument("--logreg-c", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-hidden-dim", type=int, default=192)
    parser.add_argument("--torch-depth", type=int, default=1)
    parser.add_argument("--torch-dropout", type=float, default=0.10)
    parser.add_argument("--torch-epochs", type=int, default=120)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-lr", type=float, default=2e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-3)
    parser.add_argument("--torch-seed", type=int, default=20260522)
    parser.add_argument("--torch-class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--torch-pca-components", type=int, default=160)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-margin-column", choices=ATTR_COLUMNS, default="effectiveness")
    parser.add_argument("--write-effectiveness-margin", type=float)
    parser.add_argument(
        "--write-variant",
        choices=["punch_type", "effectiveness", "ptype_eff", "hand_target", "all_attrs"],
        default="effectiveness",
    )
    parser.add_argument("--guard-from", default="")
    parser.add_argument("--guard-to", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
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
    print_score("baseline", baseline, len(pred_rows), 0)

    features, matched = load_aligned_features(args, pred_rows, video_by_key)
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    train_columns = split_columns(args.train_columns)
    margin_columns = split_columns(args.margin_columns)
    print(
        f"videomae_features={features.shape} rows={len(pred_rows)} matched={int(matched.sum())} "
        f"pooling={args.pooling} row_context={not args.no_row_context} "
        f"head={args.head_type} train_columns={','.join(train_columns)}",
        flush=True,
    )

    predictions, probabilities = oof_attribute_predictions(
        args,
        features,
        labels,
        groups,
        pred_rows,
        matched,
        train_columns,
    )
    variants = {
        "punch_type": ["punch_type"],
        "effectiveness": ["effectiveness"],
        "ptype_eff": ["punch_type", "effectiveness"],
        "hand_target": ["hand", "target"],
        "all_attrs": ATTR_COLUMNS,
    }
    rows_by_variant = {}
    print("variant,score,delta,type,effectiveness,hand,target,wins,n_changed")
    for name, columns in variants.items():
        rows, changed = apply_predictions(pred_rows, predictions, columns)
        rows_by_variant[name] = rows
        print_variant(name, rows, gt_rows, baseline, changed)
    for column in margin_columns:
        if column not in ATTR_COLUMNS:
            raise ValueError(f"unknown margin column {column}")
        for margin in parse_floats(args.effectiveness_margins):
            name = f"{column}_margin_{margin:g}"
            rows, changed = apply_attribute_margin(pred_rows, predictions, probabilities, column, margin)
            rows, guarded_changed = apply_transition_guard(args, pred_rows, rows, column)
            rows_by_variant[name] = rows
            print_variant(name, rows, gt_rows, baseline, guarded_changed if has_guard(args) else changed)

    if args.write_oof_rows:
        variant_name = args.write_variant
        if args.write_effectiveness_margin is not None:
            variant_name = f"{args.write_margin_column}_margin_{args.write_effectiveness_margin:g}"
        if variant_name not in rows_by_variant:
            raise KeyError(f"variant {variant_name} was not evaluated")
        write_csv_rows(args.write_oof_rows, rows_by_variant[variant_name], SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} variant={variant_name}", flush=True)
    print_label_counts(labels)
    return 0


def load_aligned_features(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(args.feature_cache)
    raw_features = cache["features"].astype(np.float32)
    index_rows = read_csv_rows(resolve_feature_index(args))
    if len(index_rows) != len(raw_features):
        raise ValueError(f"index rows {len(index_rows)} != feature rows {len(raw_features)}")
    feature_by_key = {
        (row["video_key"], row["id"]): raw_features[int(row["feature_index"])]
        for row in index_rows
    }
    sample = aggregate_feature(raw_features[0], args.pooling)
    if not args.no_row_context:
        sample = np.concatenate([sample, row_context_features(rows[0], video_by_key[rows[0]["video_key"]])])
    output = np.zeros((len(rows), len(sample)), dtype=np.float32)
    matched = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        key = (row["video_key"], row["id"])
        if key not in feature_by_key:
            continue
        vector = aggregate_feature(feature_by_key[key], args.pooling)
        if not args.no_row_context:
            vector = np.concatenate([vector, row_context_features(row, video_by_key[row["video_key"]])])
        output[index] = vector.astype(np.float32)
        matched[index] = True
    return output, matched


def resolve_feature_index(args: argparse.Namespace) -> Path:
    if args.feature_index is not None:
        return args.feature_index
    text = str(args.feature_cache)
    if not text.endswith(".npz"):
        raise ValueError("Cannot infer feature index from a non-.npz feature cache")
    return Path(text[:-4] + "_index.csv")


def aggregate_feature(feature: np.ndarray, pooling: str) -> np.ndarray:
    if feature.ndim == 1:
        return feature.astype(np.float32)
    if feature.ndim != 2:
        raise ValueError(f"expected [T,D] feature, got {feature.shape}")
    if pooling == "mean":
        return feature.mean(axis=0).astype(np.float32)
    if pooling == "flatten":
        return feature.reshape(-1).astype(np.float32)
    return np.concatenate([feature.mean(axis=0), feature.std(axis=0)]).astype(np.float32)


def apply_transition_guard(
    args: argparse.Namespace,
    original_rows: list[dict[str, str]],
    modified_rows: list[dict[str, str]],
    column: str,
) -> tuple[list[dict[str, str]], int]:
    from_values = split_values(args.guard_from)
    to_values = split_values(args.guard_to)
    if not from_values and not to_values:
        changed = sum(new[column] != old[column] for old, new in zip(original_rows, modified_rows))
        return modified_rows, changed
    output = []
    changed = 0
    for old, new in zip(original_rows, modified_rows):
        item = dict(new)
        if item[column] != old[column]:
            allowed_from = not from_values or old[column] in from_values
            allowed_to = not to_values or item[column] in to_values
            if allowed_from and allowed_to:
                changed += 1
            else:
                item[column] = old[column]
        output.append(item)
    return output, changed


def has_guard(args: argparse.Namespace) -> bool:
    return bool(split_values(args.guard_from) or split_values(args.guard_to))


def print_variant(
    name: str,
    rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    baseline: dict[str, object],
    changed: int,
) -> None:
    score = score_predictions(gt_rows, rows)
    summary = attr_summary(score)
    print(
        f"{name},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:.6f},"
        f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
        f"{summary['hand']:.6f},{summary['target']:.6f},"
        f"{video_wins(score, baseline)},{changed}",
        flush=True,
    )


def split_columns(text: str) -> list[str]:
    columns = [value.strip() for value in text.split(",") if value.strip()]
    for column in columns:
        if column not in ATTR_COLUMNS:
            raise ValueError(f"unknown attribute column {column}")
    return columns


def split_values(text: str) -> set[str]:
    return {value.strip() for value in text.split(",") if value.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
