#!/usr/bin/env python3
"""Train fixed-row attribute models on validation rows and apply to a submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_exchange_state_gate import index_candidates, parse_ints
from tools.evaluate_fixed_row_attribute_model import (
    ATTR_COLUMNS,
    apply_predictions,
    attribute_features,
    matched_attribute_labels,
)


VARIANTS = {
    "punch_type": ["punch_type"],
    "effectiveness": ["effectiveness"],
    "ptype_eff": ["punch_type", "effectiveness"],
    "hand_target": ["hand", "target"],
    "all_attrs": ATTR_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="all_attrs")
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument("--label-window", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    windows = parse_ints(args.feature_windows)
    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    train_keys = sorted({row["video_key"] for row in train_rows})
    train_indexes = load_candidate_indexes(train_keys, args.train_tracks_dir, config)
    x_train = np.stack(
        [
            attribute_features(
                row,
                train_video_by_key[row["video_key"]],
                train_indexes[row["video_key"]],
                args.match_window,
                windows,
            )
            for row in train_rows
        ]
    )
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    labels = matched_attribute_labels(train_rows, gt_rows, args.label_window)
    trainable = np.asarray([label is not None for label in labels], dtype=bool)
    models = fit_models(x_train, labels, trainable)

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "test/videos.csv")}
    test_keys = sorted({row["video_key"] for row in clear_rows})
    test_indexes = load_candidate_indexes(test_keys, args.test_tracks_dir, config)
    x_test = np.stack(
        [
            attribute_features(
                row,
                test_video_by_key[row["video_key"]],
                test_indexes[row["video_key"]],
                args.match_window,
                windows,
            )
            for row in clear_rows
        ]
    )
    predictions = predict_attrs(models, x_test)
    modified_clear, changed = apply_predictions(clear_rows, predictions, VARIANTS[args.variant])
    modified_by_id = {row["id"]: row for row in modified_clear}
    output = [
        modified_by_id[row["id"]] if row.get("clear") == "true" and row["id"] in modified_by_id else {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for row in rows
    ]
    write_csv_rows(args.output, output, SUBMISSION_COLUMNS)
    print(
        f"train_rows={len(train_rows)} labeled={int(trainable.sum())} clear_rows={len(clear_rows)} "
        f"changed={changed} variant={args.variant} output={args.output}"
    )
    return 0


def load_candidate_indexes(keys: list[str], tracks_dir: Path, config: PoseHeuristicConfig):
    return {
        key: index_candidates(
            apply_temporal_context(
                score_pose_tracks(tracks_dir / f"{key}.jsonl", config),
                config,
            )
        )
        for key in keys
    }


def fit_models(
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    trainable: np.ndarray,
):
    models = {}
    for column in ATTR_COLUMNS:
        values = [labels[index][column] for index in np.where(trainable)[0]]  # type: ignore[index]
        encoder = LabelEncoder()
        y = encoder.fit_transform(values)
        if len(encoder.classes_) == 1:
            models[column] = (encoder, None)
            continue
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.3,
            random_state=42,
        )
        model.fit(x[trainable], y)
        models[column] = (encoder, model)
    return models


def predict_attrs(models, x: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    for column, (encoder, model) in models.items():
        if model is None:
            output[column] = np.asarray([encoder.classes_[0]] * len(x), dtype=object)
        else:
            output[column] = encoder.inverse_transform(model.predict(x)).astype(object)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
