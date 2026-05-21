#!/usr/bin/env python3
"""Evaluate learned attribute replacement while keeping fixed rows/timing."""

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
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_exchange_state_gate import index_candidates, parse_ints, row_features
from tools.evaluate_fighter_rival_flip import best_candidate
from tools.evaluate_pose_selection_variants import video_wins


ATTR_COLUMNS = ["punch_type", "effectiveness", "hand", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument(
        "--write-variant",
        choices=["punch_type", "effectiveness", "ptype_eff", "hand_target", "all_attrs"],
        default="all_attrs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, 0, len(pred_rows))

    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    indexed_by_key = {
        key: index_candidates(
            apply_temporal_context(
                score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config),
                config,
            )
        )
        for key in keys
    }
    windows = parse_ints(args.feature_windows)
    x = np.stack(
        [
            attribute_features(
                row,
                video_by_key[row["video_key"]],
                indexed_by_key[row["video_key"]],
                args.match_window,
                windows,
            )
            for row in pred_rows
        ]
    )
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    predictions = oof_attribute_predictions(x, labels, groups)

    variants = {
        "punch_type": ["punch_type"],
        "effectiveness": ["effectiveness"],
        "ptype_eff": ["punch_type", "effectiveness"],
        "hand_target": ["hand", "target"],
        "all_attrs": ATTR_COLUMNS,
    }
    print("variant,score,delta,type,effectiveness,hand,target,wins,n_changed")
    rows_by_variant = {}
    for name, columns in variants.items():
        rows, changed = apply_predictions(pred_rows, predictions, columns)
        rows_by_variant[name] = rows
        score = score_predictions(gt_rows, rows)
        summary = attr_summary(score)
        print(
            f"{name},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:.6f},"
            f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
            f"{summary['hand']:.6f},{summary['target']:.6f},"
            f"{video_wins(score, baseline)},{changed}"
        )
    if args.write_oof_rows:
        write_csv_rows(args.write_oof_rows, rows_by_variant[args.write_variant], SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} "
            f"variant={args.write_variant} n_rows={len(rows_by_variant[args.write_variant])}"
        )
    print_label_counts(labels)
    return 0


def attribute_features(
    row: dict[str, str],
    video: dict[str, str],
    candidates_by_frame,
    match_window: int,
    windows: list[int],
) -> np.ndarray:
    base = row_features(row, video, candidates_by_frame, match_window, windows).tolist()
    frame = as_int(row["frame"], "frame")
    candidate = best_candidate(
        candidates_by_frame,
        frame,
        match_window,
        lambda item: item.fighter == row["fighter"] and item.hand == row["hand"],
    )
    if candidate is None:
        candidate = best_candidate(candidates_by_frame, frame, match_window, lambda item: item.fighter == row["fighter"])
    base.extend(
        [
            1.0 if row["punch_type"] == "hook" else 0.0,
            1.0 if row["punch_type"] == "cross" else 0.0,
            1.0 if row["punch_type"] == "uppercut" else 0.0,
            1.0 if row["effectiveness"] == "landed" else 0.0,
            1.0 if row["effectiveness"] == "blocked" else 0.0,
            1.0 if row["effectiveness"] == "missed" else 0.0,
            float(candidate.score) if candidate else 0.0,
        ]
    )
    return np.asarray(base, dtype=np.float32)


def matched_attribute_labels(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    label_window: int,
) -> list[dict[str, str] | None]:
    labels: list[dict[str, str] | None] = [None] * len(pred_rows)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    global_indices_by_key: dict[str, list[int]] = {}
    for index, row in enumerate(pred_rows):
        global_indices_by_key.setdefault(row["video_key"], []).append(index)
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            pred = rows[match.pred_index]
            gt = gt_by_key[key][match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) > label_window:
                continue
            labels[global_indices_by_key[key][match.pred_index]] = gt
    return labels


def oof_attribute_predictions(
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    groups: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {column: np.asarray([""] * len(labels), dtype=object) for column in ATTR_COLUMNS}
    trainable = np.asarray([label is not None for label in labels], dtype=bool)
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = (groups != group) & trainable
        if train.sum() < 10:
            continue
        for column in ATTR_COLUMNS:
            values = [labels[index][column] for index in np.where(train)[0]]  # type: ignore[index]
            encoder = LabelEncoder()
            y = encoder.fit_transform(values)
            if len(encoder.classes_) == 1:
                output[column][valid] = encoder.classes_[0]
                continue
            model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.3,
                random_state=42,
            )
            model.fit(x[train], y)
            output[column][valid] = encoder.inverse_transform(model.predict(x[valid]))
        print(f"fold={group} train={int(train.sum())} valid={int(valid.sum())}", flush=True)
    for column in ATTR_COLUMNS:
        for index, value in enumerate(output[column]):
            if value == "":
                output[column][index] = labels[index][column] if labels[index] else ""
    return output


def apply_predictions(
    rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    columns: list[str],
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for column in columns:
            value = str(predictions[column][index])
            if value:
                item[column] = value
        if any(item[column] != row[column] for column in columns):
            changed += 1
        output.append(item)
    return output, changed


def print_score(label: str, score: dict[str, object], wins: int, n_rows: int) -> None:
    summary = attr_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},type={summary['punch_type']:.6f},"
        f"eff={summary['effectiveness']:.6f},hand={summary['hand']:.6f},"
        f"target={summary['target']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def attr_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "punch_type": float(np.mean([item["score_punch_type"] for item in by_video.values()])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
    }


def print_label_counts(labels: list[dict[str, str] | None]) -> None:
    usable = [label for label in labels if label is not None]
    print(f"labeled_rows={len(usable)}")
    for column in ATTR_COLUMNS:
        counts: dict[str, int] = {}
        for label in usable:
            counts[label[column]] = counts.get(label[column], 0) + 1
        print("labels_" + column + "=" + ",".join(f"{key}:{value}" for key, value in sorted(counts.items())))


def fight_group(video: dict[str, str]) -> str:
    return "|".join([video["dataset_type"], video["data_root"], video["fight_index"], video["fight_folder"]])


if __name__ == "__main__":
    raise SystemExit(main())
