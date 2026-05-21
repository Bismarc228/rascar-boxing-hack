#!/usr/bin/env python3
"""Evaluate RGB embeddings as a timing-offset witness for fixed rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from tools.evaluate_pose_selection_variants import score_summary, video_wins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--model", choices=["ridge", "huber"], default="ridge")
    parser.add_argument("--max-shifts", default="2,4,6,8,10,12")
    parser.add_argument("--scales", default="0.25,0.5,0.75,1.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    pred_rows = [row for row in read_csv_rows(args.predictions) if row["clear"] == "true"]
    features = np.load(args.feature_cache)["features"].astype(np.float32)
    if len(features) != len(pred_rows):
        raise ValueError(f"feature count {len(features)} != prediction count {len(pred_rows)}")
    keys = sorted({row["video_key"] for row in pred_rows})
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row["clear"] == "true" and row["video_key"] in set(keys)
    ]
    offsets = matched_offsets(gt_rows, pred_rows)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    pred_offsets = oof_offsets(args.model, features, offsets, groups)

    baseline = score_predictions(gt_rows, pred_rows)
    baseline_summary = score_summary(baseline)
    print(
        f"baseline score={baseline['macro_score']:.6f} "
        f"time={baseline_summary['time']:.6f} fp={baseline_summary['fp_penalty']:.6f} "
        f"n={len(pred_rows)} matched={int(np.isfinite(offsets).sum())}"
    )
    print("score,delta,time,fp,wins,n_rows,max_shift,scale,mean_abs_shift")
    results = []
    for max_shift in parse_ints(args.max_shifts):
        for scale in parse_floats(args.scales):
            rows = shift_rows(pred_rows, pred_offsets, max_shift, scale, video_by_key)
            score = score_predictions(gt_rows, rows)
            summary = score_summary(score)
            mean_abs_shift = float(np.mean([abs(int(a["frame"]) - int(b["frame"])) for a, b in zip(pred_rows, rows)]))
            results.append(
                (
                    score["macro_score"],
                    summary["time"],
                    summary["fp_penalty"],
                    video_wins(score, baseline),
                    len(rows),
                    max_shift,
                    scale,
                    mean_abs_shift,
                    score,
                )
            )
    for macro, time_score, fp, wins, n_rows, max_shift, scale, mean_abs_shift, _score in sorted(results, reverse=True):
        print(
            f"{macro:.6f},{macro - baseline['macro_score']:.6f},"
            f"{time_score:.6f},{fp:.6f},{wins},{n_rows},{max_shift},{scale},{mean_abs_shift:.3f}"
        )
    print_video_offset_summary(pred_rows, offsets, pred_offsets)
    return 0


def matched_offsets(gt_rows: list[dict[str, str]], pred_rows: list[dict[str, str]]) -> np.ndarray:
    output = np.full(len(pred_rows), np.nan, dtype=np.float32)
    gt_by_key = group_rows(gt_rows)
    pred_by_key = group_rows(pred_rows)
    offset = 0
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            gt_frame = int(gt_by_key[key][match.gt_index]["frame"])
            pred_frame = int(rows[match.pred_index]["frame"])
            output[offset + match.pred_index] = float(gt_frame - pred_frame)
        offset += len(rows)
    return output


def oof_offsets(
    model_name: str,
    features: np.ndarray,
    offsets: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    output = np.zeros(len(offsets), dtype=np.float32)
    trainable = np.isfinite(offsets)
    for group in sorted(set(groups.tolist())):
        train = (groups != group) & trainable
        valid = groups == group
        if train.sum() < 10:
            output[valid] = 0.0
            continue
        model = make_model(model_name)
        model.fit(features[train], offsets[train])
        output[valid] = model.predict(features[valid]).astype(np.float32)
    return output


def make_model(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    return make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, max_iter=300))


def shift_rows(
    rows: list[dict[str, str]],
    offsets: np.ndarray,
    max_shift: int,
    scale: float,
    video_by_key: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    output = []
    for row, offset in zip(rows, offsets):
        item = dict(row)
        shift = int(round(float(np.clip(offset * scale, -max_shift, max_shift))))
        frame_count = int(video_by_key[row["video_key"]]["frame_count"])
        item["frame"] = str(max(0, min(frame_count - 1, int(row["frame"]) + shift)))
        output.append(item)
    return output


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_key"]].append(row)
    return dict(grouped)


def fight_group(video: dict[str, str]) -> str:
    return f"{video['data_root']}::{video['fight_index']}"


def print_video_offset_summary(
    rows: list[dict[str, str]],
    offsets: np.ndarray,
    pred_offsets: np.ndarray,
) -> None:
    print("video_key,n,matched,true_offset_mean,true_offset_mae,pred_offset_mean,pred_offset_mae")
    start = 0
    for key, items in group_rows(rows).items():
        stop = start + len(items)
        true = offsets[start:stop]
        pred = pred_offsets[start:stop]
        mask = np.isfinite(true)
        if mask.any():
            true_mean = float(np.mean(true[mask]))
            true_mae = float(np.mean(np.abs(true[mask])))
            pred_mae = float(np.mean(np.abs(pred[mask] - true[mask])))
        else:
            true_mean = 0.0
            true_mae = 0.0
            pred_mae = 0.0
        print(
            f"{key},{len(items)},{int(mask.sum())},{true_mean:.3f},{true_mae:.3f},"
            f"{float(np.mean(pred)):.3f},{pred_mae:.3f}"
        )
        start = stop


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
