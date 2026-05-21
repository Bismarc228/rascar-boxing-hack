#!/usr/bin/env python3
"""Train exchange-side fighter flip model on validation rows and apply to test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_exchange_side_fighter_model import (
    apply_flips,
    build_features,
    flip_labels,
    index_candidates,
    make_model,
    parse_ints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="logreg")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--match-mode", choices=["same_hand_target", "same_hand", "any"], default="same_hand")
    parser.add_argument("--update-mode", choices=["fighter_only", "fighter_hand_target"], default="fighter_only")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--feature-windows", default="2,4,8")
    parser.add_argument("--protect-video-keys", default="", help="Comma-separated videos whose rows must not be flipped")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    feature_windows = parse_ints(args.feature_windows)

    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_keys = sorted({row["video_key"] for row in train_rows})
    train_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    train_indexes = load_candidate_indexes(train_keys, args.train_tracks_dir, config)
    x_train, action_train, _rivals_train = build_features(
        train_rows,
        train_video_by_key,
        train_indexes,
        args.window,
        args.match_mode,
        feature_windows,
    )
    y_train = flip_labels(train_rows, gt_rows)
    fit_mask = action_train
    if int(fit_mask.sum()) < 20 or len(np.unique(y_train[fit_mask])) < 2:
        raise SystemExit("Not enough actionable positive/negative training rows")
    model = make_model(args.model)
    model.fit(x_train[fit_mask], y_train[fit_mask])

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_keys = sorted({row["video_key"] for row in clear_rows})
    test_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "test/videos.csv")}
    test_indexes = load_candidate_indexes(test_keys, args.test_tracks_dir, config)
    x_test, action_test, rivals = build_features(
        clear_rows,
        test_video_by_key,
        test_indexes,
        args.window,
        args.match_mode,
        feature_windows,
    )
    probabilities = model.predict_proba(x_test)[:, 1].astype(np.float32)
    protected = split_keys(args.protect_video_keys)
    protect_mask = np.asarray([row["video_key"] in protected for row in clear_rows], dtype=bool)
    flip_mask = (probabilities >= args.threshold) & action_test & ~protect_mask
    modified_clear, changed = apply_flips(clear_rows, rivals, flip_mask, args.update_mode)
    modified_by_id = {row["id"]: row for row in modified_clear}

    output = []
    for row in rows:
        if row.get("clear") == "true" and row["id"] in modified_by_id:
            output.append(modified_by_id[row["id"]])
        else:
            output.append({col: row.get(col, "") for col in SUBMISSION_COLUMNS})
    write_csv_rows(args.output, output, SUBMISSION_COLUMNS)
    print(
        f"train_rows={len(train_rows)} train_actionable={int(action_train.sum())} "
        f"train_pos={int(y_train[fit_mask].sum())} clear_rows={len(clear_rows)} "
        f"actionable={int(action_test.sum())} changed={changed} threshold={args.threshold} "
        f"model={args.model} window={args.window} match_mode={args.match_mode} "
        f"update_mode={args.update_mode} protected={','.join(sorted(protected))} output={args.output}"
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


def split_keys(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
