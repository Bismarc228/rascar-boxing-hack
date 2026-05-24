#!/usr/bin/env python3
"""Train a guarded target-only correction head and apply it to test rows."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import PoseHeuristicConfig
from tools.evaluate_fixed_row_attribute_model import attribute_features, matched_attribute_labels
from tools.evaluate_target_component_guard import (
    TARGET_VALUES,
    build_guard_mask,
    build_pose_stats,
    effectiveness_predicate,
    load_candidate_indexes,
    transition_predicate_factory,
)
from tools.evaluate_exchange_state_gate import parse_ints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-tracks-dir", type=Path, action="append", required=True)
    parser.add_argument("--test-feature-tracks-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--effectiveness-mode", default="miss")
    parser.add_argument("--transition-mode", default="both")
    parser.add_argument("--min-prob", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=0.25)
    parser.add_argument("--min-stable-votes", type=int, default=2)
    parser.add_argument("--require-pose-majority", action="store_true")
    parser.add_argument("--require-pose-change", action="store_true")
    parser.add_argument("--public-sensitive-videos", default="agn_037,agn_038,agn_039")
    parser.add_argument("--public-policy", choices=["all", "exclude_public"], default="all")
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
    target_pred, target_prob = fit_and_predict_target(args, config, windows)

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in clear_rows})
    pose_indexes = [load_candidate_indexes(keys, tracks_dir, config) for tracks_dir in args.test_tracks_dir]
    pose_stats = build_pose_stats(clear_rows, pose_indexes, args.match_window)
    public_sensitive = split_values(args.public_sensitive_videos)
    mask = build_guard_mask(
        clear_rows,
        target_pred,
        target_prob,
        pose_stats,
        effectiveness_predicate(args.effectiveness_mode),
        transition_predicate_factory(args.transition_mode),
        args.min_prob,
        args.min_margin,
        args.min_stable_votes,
        args.require_pose_majority,
        args.require_pose_change,
        args.public_policy,
        public_sensitive,
    )
    modified_clear, changed = apply_target_mask(clear_rows, target_pred, mask)
    modified_by_id = {row["id"]: row for row in modified_clear}
    output = [
        modified_by_id[row["id"]]
        if row.get("clear") == "true" and row["id"] in modified_by_id
        else {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for row in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output, output, SUBMISSION_COLUMNS)
    print_summary(args, clear_rows, modified_clear, changed, public_sensitive)
    return 0


def fit_and_predict_target(
    args: argparse.Namespace,
    config: PoseHeuristicConfig,
    windows: list[int],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_keys = sorted({row["video_key"] for row in train_rows})
    train_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    train_gt = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
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
    labels = matched_attribute_labels(train_rows, train_gt, args.label_window)
    trainable = np.asarray([label is not None for label in labels], dtype=bool)
    if int(trainable.sum()) < 20:
        raise SystemExit(f"too few labeled train rows: {int(trainable.sum())}")

    encoder = LabelEncoder()
    y = encoder.fit_transform([labels[index]["target"] for index in np.where(trainable)[0]])  # type: ignore[index]
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.3,
        random_state=42,
    )
    model.fit(x_train[trainable], y)

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_keys = sorted({row["video_key"] for row in clear_rows})
    test_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "test/videos.csv")}
    feature_tracks_dir = args.test_feature_tracks_dir or args.test_tracks_dir[0]
    test_indexes = load_candidate_indexes(test_keys, feature_tracks_dir, config)
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
    encoded = model.predict(x_test)
    proba = model.predict_proba(x_test)
    predictions = encoder.inverse_transform(encoded).astype(object)
    probabilities = []
    for row_proba in proba:
        item = {value: 0.0 for value in TARGET_VALUES}
        item.update({str(name): float(value) for name, value in zip(encoder.classes_, row_proba)})
        probabilities.append(item)
    print(
        f"train_rows={len(train_rows)} labeled={int(trainable.sum())} "
        f"clear_rows={len(clear_rows)} target_labels={format_counts(Counter(encoder.inverse_transform(y)))}"
    )
    return predictions, probabilities


def apply_target_mask(
    rows: list[dict[str, str]],
    target_pred: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if mask[index] and str(target_pred[index]):
            item["target"] = str(target_pred[index])
        if item["target"] != row["target"]:
            changed += 1
        output.append(item)
    return output, changed


def print_summary(
    args: argparse.Namespace,
    original_rows: list[dict[str, str]],
    modified_rows: list[dict[str, str]],
    changed: int,
    public_sensitive: set[str],
) -> None:
    by_video: Counter[str] = Counter()
    by_effectiveness: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    public_changed = 0
    for old, new in zip(original_rows, modified_rows):
        if old["target"] == new["target"]:
            continue
        by_video[old["video_key"]] += 1
        by_effectiveness[old["effectiveness"]] += 1
        transitions[f"{old['target']}->{new['target']}"] += 1
        if old["video_key"] in public_sensitive:
            public_changed += 1
    print(f"output={args.output}")
    print(f"changed={changed}")
    print(f"public_changed={public_changed}")
    print(f"by_video={format_counts(by_video)}")
    print(f"by_effectiveness={format_counts(by_effectiveness)}")
    print(f"transitions={format_counts(transitions)}")
    print(
        "guard="
        f"effectiveness:{args.effectiveness_mode},transition:{args.transition_mode},"
        f"min_prob:{args.min_prob},min_margin:{args.min_margin},"
        f"min_stable_votes:{args.min_stable_votes},"
        f"require_pose_majority:{args.require_pose_majority},"
        f"require_pose_change:{args.require_pose_change},"
        f"public_policy:{args.public_policy}"
    )


def split_values(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def format_counts(counts: Counter[str]) -> str:
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


if __name__ == "__main__":
    raise SystemExit(main())
