#!/usr/bin/env python3
"""Train a frozen VideoMAE fixed-row attribute head and apply it to a submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from tools.evaluate_rgb_fixed_row_attribute_model import (
    ATTR_COLUMNS,
    apply_attribute_margin,
    fit_torch_mlp_classifier,
    matched_attribute_labels,
    resolve_device,
)
from tools.evaluate_videomae_fixed_row_attribute_model import (
    aggregate_feature,
    resolve_feature_index,
    row_context_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-feature-cache", type=Path, required=True)
    parser.add_argument("--train-feature-index", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-feature-cache", type=Path, required=True)
    parser.add_argument("--test-feature-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column", choices=ATTR_COLUMNS, default="effectiveness")
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--guard-from", default="")
    parser.add_argument("--guard-to", default="")
    parser.add_argument("--protect-video-keys", default="")
    parser.add_argument("--pooling", choices=["meanstd", "mean", "flatten"], default="flatten")
    parser.add_argument("--no-row-context", action="store_true")
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--head-type", choices=["logreg", "torch_linear", "torch_mlp"], default="torch_mlp")
    parser.add_argument("--pca-components", type=int, default=160)
    parser.add_argument("--logreg-c", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-hidden-dim", type=int, default=192)
    parser.add_argument("--torch-depth", type=int, default=1)
    parser.add_argument("--torch-dropout", type=float, default=0.10)
    parser.add_argument("--torch-epochs", type=int, default=80)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-lr", type=float, default=2e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-3)
    parser.add_argument("--torch-seed", type=int, default=20260522)
    parser.add_argument("--torch-class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--torch-pca-components", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    train_keys = sorted({row["video_key"] for row in train_rows})
    missing_train = sorted(set(train_keys) - set(train_video_by_key))
    if missing_train:
        raise RuntimeError(f"Training prediction keys are not train videos: {missing_train}")
    x_train, matched_train = load_aligned_features(
        args,
        train_rows,
        train_video_by_key,
        args.train_feature_cache,
        args.train_feature_index,
    )
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    labels = matched_attribute_labels(train_rows, gt_rows, args.label_window)
    trainable = np.asarray([label is not None for label in labels], dtype=bool) & matched_train
    if int(trainable.sum()) < 20:
        raise RuntimeError(f"Not enough labeled rows: {int(trainable.sum())}")

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "test/videos.csv")}
    test_keys = sorted({row["video_key"] for row in clear_rows})
    missing_test = sorted(set(test_keys) - set(test_video_by_key))
    if missing_test:
        raise RuntimeError(f"Input keys are not test videos: {missing_test}")
    x_test, matched_test = load_aligned_features(
        args,
        clear_rows,
        test_video_by_key,
        args.test_feature_cache,
        args.test_feature_index,
    )
    predictions, probabilities = fit_predict(args, x_train, labels, trainable, x_test, matched_test, clear_rows)
    pred_map = {column: np.asarray([row[column] for row in clear_rows], dtype=object) for column in ATTR_COLUMNS}
    prob_map = {column: [{row[column]: 1.0} for row in clear_rows] for column in ATTR_COLUMNS}
    pred_map[args.column] = predictions
    prob_map[args.column] = probabilities
    modified_clear, changed_before_guard = apply_attribute_margin(
        clear_rows,
        pred_map,
        prob_map,
        args.column,
        args.margin,
    )
    modified_clear, changed_after_guard = apply_transition_guard(args, clear_rows, modified_clear)

    protected = split_values(args.protect_video_keys)
    modified_by_id = {}
    changed = 0
    protected_changed = 0
    for original, modified in zip(clear_rows, modified_clear):
        if original["video_key"] in protected:
            item = {col: original.get(col, "") for col in SUBMISSION_COLUMNS}
            if modified[args.column] != original[args.column]:
                protected_changed += 1
        else:
            item = modified
            if item[args.column] != original[args.column]:
                changed += 1
        modified_by_id[original["id"]] = item

    output = [
        modified_by_id[row["id"]]
        if row.get("clear") == "true" and row["id"] in modified_by_id
        else {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for row in rows
    ]
    write_csv_rows(args.output, output, SUBMISSION_COLUMNS)
    print(
        f"train_rows={len(train_rows)} trainable={int(trainable.sum())} "
        f"test_clear_rows={len(clear_rows)} matched_test={int(matched_test.sum())} "
        f"column={args.column} margin={args.margin:g} "
        f"changed_before_guard={changed_before_guard} changed_after_guard={changed_after_guard} "
        f"changed={changed} protected_changed={protected_changed} protected={','.join(sorted(protected))} "
        f"head={args.head_type} pooling={args.pooling} device={resolve_device(args.device)} output={args.output}",
        flush=True,
    )
    return 0


def load_aligned_features(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    feature_cache: Path,
    feature_index: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(feature_cache)
    raw_features = cache["features"].astype(np.float32)
    index_args = argparse.Namespace(feature_cache=feature_cache, feature_index=feature_index)
    index_rows = read_csv_rows(resolve_feature_index(index_args))
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


def fit_predict(
    args: argparse.Namespace,
    x_train: np.ndarray,
    labels: list[dict[str, str] | None],
    trainable: np.ndarray,
    x_test: np.ndarray,
    matched_test: np.ndarray,
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    predictions = np.asarray([row[args.column] for row in rows], dtype=object)
    probabilities = [{row[args.column]: 1.0} for row in rows]
    predict_indices = np.where(matched_test)[0]
    if len(predict_indices) == 0:
        return predictions, probabilities
    values = [labels[index][args.column] for index in np.where(trainable)[0]]  # type: ignore[index]
    encoder = LabelEncoder()
    y = encoder.fit_transform(values)
    if len(encoder.classes_) == 1:
        value = str(encoder.classes_[0])
        predictions[predict_indices] = value
        for index in predict_indices:
            probabilities[index] = {value: 1.0}
        return predictions, probabilities
    if args.head_type.startswith("torch_"):
        pred_encoded, proba = fit_torch_mlp_classifier(
            args,
            x_train[trainable],
            y,
            x_test[predict_indices],
            len(encoder.classes_),
            args.torch_seed,
        )
    else:
        train_x = x_train[trainable]
        n_components = min(args.pca_components, train_x.shape[0] - 1, train_x.shape[1])
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=42),
            LogisticRegression(C=args.logreg_c, max_iter=1200, class_weight="balanced", random_state=42),
        )
        model.fit(train_x, y)
        pred_encoded = model.predict(x_test[predict_indices])
        proba = model.predict_proba(x_test[predict_indices])
    pred_values = encoder.inverse_transform(pred_encoded)
    class_values = [str(value) for value in encoder.classes_]
    predictions[predict_indices] = pred_values
    for index, row_probs in zip(predict_indices, proba):
        probabilities[index] = {
            class_value: float(prob)
            for class_value, prob in zip(class_values, row_probs)
        }
    return predictions, probabilities


def apply_transition_guard(
    args: argparse.Namespace,
    original_rows: list[dict[str, str]],
    modified_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    from_values = split_values(args.guard_from)
    to_values = split_values(args.guard_to)
    if not from_values and not to_values:
        changed = sum(new[args.column] != old[args.column] for old, new in zip(original_rows, modified_rows))
        return modified_rows, changed
    output = []
    changed = 0
    for old, new in zip(original_rows, modified_rows):
        item = dict(new)
        if item[args.column] != old[args.column]:
            allowed_from = not from_values or old[args.column] in from_values
            allowed_to = not to_values or item[args.column] in to_values
            if allowed_from and allowed_to:
                changed += 1
            else:
                item[args.column] = old[args.column]
        output.append(item)
    return output, changed


def split_values(text: str) -> set[str]:
    return {value.strip() for value in text.split(",") if value.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
