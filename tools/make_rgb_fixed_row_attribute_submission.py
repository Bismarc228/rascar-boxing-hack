#!/usr/bin/env python3
"""Train cached RGB fixed-row attribute models and apply them to a submission."""

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

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, PunchCandidate, score_pose_tracks, select_candidates
from tools.evaluate_rgb_contact_clip import load_or_extract_features
from tools.evaluate_rgb_fixed_row_attribute_model import (
    ATTR_COLUMNS,
    apply_attribute_margin,
    build_features,
    load_rgb_features,
    matched_attribute_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--train-rgb-feature-cache", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-tracks-dir", type=Path, required=True)
    parser.add_argument("--test-rgb-feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column", choices=ATTR_COLUMNS, default="effectiveness")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=1800)
    parser.add_argument("--candidate-match-window", type=int, default=6)
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--logreg-c", type=float, default=0.35)
    parser.add_argument("--model-name", default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--crop-modes", default="union")
    parser.add_argument("--crop-expand", type=float, default=0.16)
    parser.add_argument("--decode-mode", choices=["sequential", "seek"], default="sequential")
    parser.add_argument("--protect-video-keys", default="", help="Comma-separated videos whose rows must stay unchanged")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_keys = sorted({row["video_key"] for row in train_rows})
    train_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    train_missing = sorted(set(train_keys) - set(train_video_by_key))
    if train_missing:
        raise RuntimeError(f"Training prediction keys are not train videos: {train_missing}")

    train_candidates, train_features = load_candidate_feature_index(
        args,
        train_keys,
        train_video_by_key,
        args.train_tracks_dir,
        args.train_rgb_feature_cache,
    )
    x_train, matched_train = build_features(args, train_rows, train_video_by_key, train_candidates, train_features)
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    labels = matched_attribute_labels(train_rows, gt_rows, args.label_window)
    trainable = np.asarray([label is not None for label in labels], dtype=bool) & matched_train
    model = fit_attribute_model(args, x_train, labels, trainable, args.column)

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_keys = sorted({row["video_key"] for row in clear_rows})
    test_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "test/videos.csv")}
    test_missing = sorted(set(test_keys) - set(test_video_by_key))
    if test_missing:
        raise RuntimeError(f"Input keys are not test videos: {test_missing}")
    test_candidates, test_features = load_candidate_feature_index(
        args,
        test_keys,
        test_video_by_key,
        args.test_tracks_dir,
        args.test_rgb_feature_cache,
    )
    x_test, matched_test = build_features(args, clear_rows, test_video_by_key, test_candidates, test_features)
    predictions, probabilities = predict_attribute(model, x_test, matched_test, clear_rows, args.column)
    modified_clear, _changed_before_protection = apply_attribute_margin(
        clear_rows,
        predictions,
        probabilities,
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
        f"test_clear_rows={len(clear_rows)} matched_test={int(matched_test.sum())} "
        f"column={args.column} margin={args.margin:g} changed={changed} "
        f"protected_changed={protected_changed} protected={','.join(sorted(protected))} "
        f"output={args.output}",
        flush=True,
    )
    return 0


def load_candidate_feature_index(
    args: argparse.Namespace,
    keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    tracks_dir: Path,
    feature_cache: Path,
) -> tuple[dict[str, list[PunchCandidate]], dict[str, np.ndarray]]:
    candidates_by_key = build_candidate_pool(args, keys, tracks_dir)
    candidate_rows = candidate_feature_rows(candidates_by_key, video_by_key)
    if feature_cache.exists():
        features = load_rgb_features(feature_cache)
    else:
        cache_args = SimpleNamespace(
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
            crop_modes=args.crop_modes,
            crop_expand=args.crop_expand,
            decode_mode=args.decode_mode,
            quiet=args.quiet,
        )
        features = load_or_extract_features(cache_args, candidate_rows, video_by_key)
    if len(features) != len(candidate_rows):
        raise ValueError(f"feature count {len(features)} != candidate count {len(candidate_rows)}")
    feature_by_key: dict[str, np.ndarray] = {}
    offset = 0
    for key in keys:
        n_items = len(candidates_by_key[key])
        feature_by_key[key] = features[offset : offset + n_items]
        offset += n_items
    return candidates_by_key, feature_by_key


def build_candidate_pool(
    args: argparse.Namespace,
    keys: list[str],
    tracks_dir: Path,
) -> dict[str, list[PunchCandidate]]:
    output = {}
    for key in keys:
        raw = score_pose_tracks(tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        output[key] = select_candidates(
            raw,
            PoseHeuristicConfig(
                min_score=args.pool_min_score,
                nms_frames=args.pool_nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=0,
            ),
            args.max_candidates_per_video,
        )
    return output


def candidate_feature_rows(
    candidates_by_key: dict[str, list[PunchCandidate]],
    video_by_key: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for key in sorted(candidates_by_key):
        video = video_by_key[key]
        for index, candidate in enumerate(candidates_by_key[key], start=len(rows) + 1):
            rows.append(
                {
                    "id": str(index),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": candidate.video_key,
                    "frame": str(candidate.frame),
                    "fighter": candidate.fighter,
                    "hand": candidate.hand,
                    "target": candidate.target,
                    "punch_type": "",
                    "effectiveness": "",
                    "clear": "true",
                }
            )
    return rows


def fit_attribute_model(
    args: argparse.Namespace,
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    trainable: np.ndarray,
    column: str,
) -> tuple[LabelEncoder, object | None]:
    if int(trainable.sum()) < 20:
        raise RuntimeError(f"Not enough trainable rows for {column}: {int(trainable.sum())}")
    values = [labels[index][column] for index in np.where(trainable)[0]]  # type: ignore[index]
    encoder = LabelEncoder()
    y = encoder.fit_transform(values)
    if len(encoder.classes_) == 1:
        return encoder, None
    n_components = min(args.pca_components, int(trainable.sum()) - 1, x.shape[1])
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
    model.fit(x[trainable], y)
    return encoder, model


def predict_attribute(
    model_bundle: tuple[LabelEncoder, object | None],
    x: np.ndarray,
    matched: np.ndarray,
    rows: list[dict[str, str]],
    column: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, float]]]]:
    encoder, model = model_bundle
    predictions = {name: np.asarray([row[name] for row in rows], dtype=object) for name in ATTR_COLUMNS}
    probabilities = {
        name: [{row[name]: 1.0} for row in rows]
        for name in ATTR_COLUMNS
    }
    predict_mask = matched
    if int(predict_mask.sum()) == 0:
        return predictions, probabilities
    indices = np.where(predict_mask)[0]
    if model is None:
        value = str(encoder.classes_[0])
        predictions[column][indices] = value
        for index in indices:
            probabilities[column][index] = {value: 1.0}
        return predictions, probabilities

    pred_encoded = model.predict(x[predict_mask])
    pred_values = encoder.inverse_transform(pred_encoded)
    predictions[column][indices] = pred_values
    proba = model.predict_proba(x[predict_mask])
    class_values = [str(value) for value in encoder.classes_]
    for index, row_probs in zip(indices, proba):
        probabilities[column][index] = {
            class_value: float(prob)
            for class_value, prob in zip(class_values, row_probs)
        }
    return predictions, probabilities


def split_keys(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
