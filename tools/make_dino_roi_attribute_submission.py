#!/usr/bin/env python3
"""Train frozen DINO ROI attribute head and apply it to a submission."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
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
from tools.evaluate_dino_roi_attribute_model import build_event_features, load_or_extract_features
from tools.evaluate_rgb_fixed_row_attribute_model import (
    ATTR_COLUMNS,
    apply_attribute_margin,
    fit_torch_mlp_classifier,
    matched_attribute_labels,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--train-feature-cache", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-tracks-dir", type=Path, required=True)
    parser.add_argument("--test-feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column", choices=ATTR_COLUMNS, default="effectiveness")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--crop-expand", type=float, default=0.22)
    parser.add_argument("--roi-radius", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--head-type", choices=["torch_linear", "torch_mlp", "logreg"], default="torch_linear")
    parser.add_argument("--pca-components", type=int, default=160)
    parser.add_argument("--logreg-c", type=float, default=0.25)
    parser.add_argument("--torch-hidden-dim", type=int, default=192)
    parser.add_argument("--torch-depth", type=int, default=1)
    parser.add_argument("--torch-dropout", type=float, default=0.10)
    parser.add_argument("--torch-epochs", type=int, default=80)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-lr", type=float, default=2e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-3)
    parser.add_argument("--torch-seed", type=int, default=20260522)
    parser.add_argument("--torch-class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--torch-pca-components", type=int, default=128)
    parser.add_argument("--protect-video-keys", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    train_keys = sorted({row["video_key"] for row in train_rows})
    missing_train = sorted(set(train_keys) - set(train_video_by_key))
    if missing_train:
        raise RuntimeError(f"Training prediction keys are not train videos: {missing_train}")
    train_features = load_or_extract_features(
        feature_args(args, args.train_tracks_dir, args.train_feature_cache),
        train_rows,
        train_video_by_key,
    )
    x_train = build_event_features(train_features, train_rows, train_video_by_key)
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    labels = matched_attribute_labels(train_rows, gt_rows, args.label_window)
    trainable = np.asarray([label is not None for label in labels], dtype=bool)
    if trainable.sum() < 20:
        raise RuntimeError(f"Not enough labeled rows: {int(trainable.sum())}")

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "test/videos.csv")}
    test_keys = sorted({row["video_key"] for row in clear_rows})
    missing_test = sorted(set(test_keys) - set(test_video_by_key))
    if missing_test:
        raise RuntimeError(f"Input keys are not test videos: {missing_test}")
    test_features = load_or_extract_features(
        feature_args(args, args.test_tracks_dir, args.test_feature_cache),
        clear_rows,
        test_video_by_key,
    )
    x_test = build_event_features(test_features, clear_rows, test_video_by_key)

    predictions, probabilities = fit_predict(args, x_train, labels, trainable, x_test)
    pred_map = {column: np.asarray([row[column] for row in clear_rows], dtype=object) for column in ATTR_COLUMNS}
    prob_map = {column: [{row[column]: 1.0} for row in clear_rows] for column in ATTR_COLUMNS}
    pred_map[args.column] = predictions
    prob_map[args.column] = probabilities
    modified_clear, changed_before_protection = apply_attribute_margin(
        clear_rows,
        pred_map,
        prob_map,
        args.column,
        args.margin,
    )

    protected = split_keys(args.protect_video_keys)
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
        f"test_clear_rows={len(clear_rows)} column={args.column} margin={args.margin:g} "
        f"changed_before_protection={changed_before_protection} changed={changed} "
        f"protected_changed={protected_changed} protected={','.join(sorted(protected))} "
        f"head={args.head_type} device={resolve_device(args.device)} output={args.output}",
        flush=True,
    )
    return 0


def feature_args(args: argparse.Namespace, tracks_dir: Path, feature_cache: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=args.data_root,
        tracks_dir=tracks_dir,
        feature_cache=feature_cache,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        crop_expand=args.crop_expand,
        roi_radius=args.roi_radius,
        cpu_threads=args.cpu_threads,
        quiet=args.quiet,
    )


def fit_predict(
    args: argparse.Namespace,
    x_train: np.ndarray,
    labels: list[dict[str, str] | None],
    trainable: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    values = [labels[index][args.column] for index in np.where(trainable)[0]]  # type: ignore[index]
    encoder = LabelEncoder()
    y = encoder.fit_transform(values)
    train_x = x_train[trainable]
    if len(encoder.classes_) == 1:
        pred = np.asarray([encoder.classes_[0]] * len(x_test), dtype=object)
        probs = [{str(encoder.classes_[0]): 1.0} for _ in range(len(x_test))]
        return pred, probs
    if args.head_type.startswith("torch_"):
        pred_encoded, proba = fit_torch_mlp_classifier(
            args,
            train_x,
            y,
            x_test,
            len(encoder.classes_),
            args.torch_seed,
        )
    else:
        n_components = min(args.pca_components, train_x.shape[0] - 1, train_x.shape[1])
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=42),
            LogisticRegression(C=args.logreg_c, max_iter=1200, class_weight="balanced", random_state=42),
        )
        model.fit(train_x, y)
        pred_encoded = model.predict(x_test)
        proba = model.predict_proba(x_test)
    class_values = [str(value) for value in encoder.classes_]
    pred_values = encoder.inverse_transform(pred_encoded)
    probabilities = [
        {class_value: float(prob) for class_value, prob in zip(class_values, row_probs)}
        for row_probs in proba
    ]
    return pred_values.astype(object), probabilities


def split_keys(text: str) -> set[str]:
    return {value.strip() for value in text.split(",") if value.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
