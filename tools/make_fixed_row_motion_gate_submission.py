#!/usr/bin/env python3
"""Train crop-motion fixed-row gate on validation rows and apply to a submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from tools.evaluate_exchange_state_gate import (
    build_labels,
    fit_model,
    index_candidates,
    parse_ints,
)
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_fixed_row_motion_gate import combined_features, compute_all_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument("--motion-offsets", default="-4,-2,0,2,4")
    parser.add_argument("--resize-width", type=int, default=240)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    offsets = parse_ints(args.motion_offsets)
    windows = parse_ints(args.feature_windows)

    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    train_keys = sorted({row["video_key"] for row in train_rows})
    train_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    train_indexes = load_candidate_indexes(train_keys, args.train_tracks_dir, config)
    train_motion = compute_all_motion(
        SimpleNamespace(
            data_root=args.data_root,
            tracks_dir=args.train_tracks_dir,
            resize_width=args.resize_width,
            quiet=args.quiet,
            jobs=args.jobs,
        ),
        train_rows,
        train_video_by_key,
        offsets,
    )
    x_train = np.stack(
        [
            combined_features(
                row,
                train_video_by_key[row["video_key"]],
                train_indexes[row["video_key"]],
                train_motion[row["video_key"]],
                offsets,
                args.match_window,
                windows,
            )
            for row in train_rows
        ]
    )
    y_train, labels = build_labels(train_rows, gt_rows)
    model = fit_model(args.model, x_train, y_train)

    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    test_keys = sorted({row["video_key"] for row in clear_rows})
    test_video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "test/videos.csv")}
    test_indexes = load_candidate_indexes(test_keys, args.test_tracks_dir, config)
    test_motion = compute_all_motion(
        SimpleNamespace(
            data_root=args.data_root,
            tracks_dir=args.test_tracks_dir,
            resize_width=args.resize_width,
            quiet=args.quiet,
            jobs=args.jobs,
        ),
        clear_rows,
        test_video_by_key,
        offsets,
    )
    x_test = np.stack(
        [
            combined_features(
                row,
                test_video_by_key[row["video_key"]],
                test_indexes[row["video_key"]],
                test_motion[row["video_key"]],
                offsets,
                args.match_window,
                windows,
            )
            for row in clear_rows
        ]
    )
    p_keep = model.predict_proba(x_test)[:, 1]
    keep_by_id = {
        row["id"]: bool(prob >= args.threshold)
        for row, prob in zip(clear_rows, p_keep)
    }
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
        f"train_rows={len(train_rows)} train_pos={int(y_train.sum())} "
        f"labels=tp_scorable:{labels.count('tp_scorable')},tp_time_only:{labels.count('tp_time_only')},fp:{labels.count('fp')} "
        f"clear_rows={len(clear_rows)} dropped={dropped} threshold={args.threshold} output={args.output}"
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


if __name__ == "__main__":
    raise SystemExit(main())
