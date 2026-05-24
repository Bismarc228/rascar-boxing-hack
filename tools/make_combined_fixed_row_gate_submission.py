#!/usr/bin/env python3
"""Train combined fixed-row keep/drop gates and apply them to a submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from tools.evaluate_combined_fixed_row_gate import (
    build_pose_features,
    ensemble_keep_mask,
    fit_model,
    load_feature_caches,
)
from tools.evaluate_vit_fixed_row_attribute_head import fight_group
from tools.evaluate_vit_fixed_row_gate import build_labels, labels_to_keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--train-feature-caches", type=Path, nargs="+", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-tracks-dir", type=Path, required=True)
    parser.add_argument("--test-feature-caches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-positive", choices=["scorable", "matched"], default="scorable")
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument("--ensemble-mode", default="drop_if_both_low")
    parser.add_argument("--left-model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--right-model", choices=["hgb", "logreg"], default="logreg")
    parser.add_argument("--left-threshold", type=float, default=0.9)
    parser.add_argument("--right-threshold", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_videos = read_csv_rows(args.data_root / "train/videos.csv")
    train_video_by_key = {row["video_key"]: row for row in train_videos}
    train_keys = sorted({row["video_key"] for row in train_rows})
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    labels = build_labels(train_rows, gt_rows)
    y = labels_to_keep(labels, args.label_positive)

    train_pose_args = argparse.Namespace(
        tracks_dir=args.train_tracks_dir,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
        match_window=args.match_window,
        feature_windows=args.feature_windows,
    )
    x_train_pose = build_pose_features(train_pose_args, train_rows, train_video_by_key, train_keys)
    x_train_vit = load_feature_caches(args.train_feature_caches)
    if len(x_train_vit) != len(train_rows):
        raise ValueError(f"train vit features {len(x_train_vit)} != train rows {len(train_rows)}")
    left = fit_model(args.left_model, x_train_pose, y)
    right = fit_model(args.right_model, x_train_vit, y)

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    test_video_by_key = {row["video_key"]: row for row in test_videos}
    test_keys = sorted({row["video_key"] for row in clear_rows})
    test_pose_args = argparse.Namespace(
        tracks_dir=args.test_tracks_dir,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
        match_window=args.match_window,
        feature_windows=args.feature_windows,
    )
    x_test_pose = build_pose_features(test_pose_args, clear_rows, test_video_by_key, test_keys)
    x_test_vit = load_feature_caches(args.test_feature_caches)
    if len(x_test_vit) != len(clear_rows):
        raise ValueError(f"test vit features {len(x_test_vit)} != clear rows {len(clear_rows)}")
    p_left = left.predict_proba(x_test_pose)[:, 1].astype(np.float32)
    p_right = right.predict_proba(x_test_vit)[:, 1].astype(np.float32)
    keep_mask = ensemble_keep_mask(
        args.ensemble_mode,
        p_left,
        p_right,
        args.left_threshold,
        args.right_threshold,
    )
    keep_by_id = {row["id"]: bool(keep) for row, keep in zip(clear_rows, keep_mask)}
    output = []
    dropped = 0
    for row in rows:
        out = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if out.get("clear") == "true" and not keep_by_id.get(out["id"], True):
            out["clear"] = "false"
            dropped += 1
        output.append(out)
    write_csv_rows(args.output, output, SUBMISSION_COLUMNS)
    print(
        f"train_rows={len(train_rows)} train_pos={int(y.sum())} clear_rows={len(clear_rows)} "
        f"dropped={dropped} mode={args.ensemble_mode} left={args.left_model}:{args.left_threshold:g} "
        f"right={args.right_model}:{args.right_threshold:g} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
