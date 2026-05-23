"""Local implementation of the challenge metric described in GOAL.md."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import EFFECTIVENESS_METRIC_MAP, FPS
from .io import as_int, group_by


@dataclass(frozen=True)
class Match:
    gt_index: int
    pred_index: int


def score_predictions(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
) -> dict[str, Any]:
    gt_by_video = group_by([row for row in gt_rows if row.get("clear") == "true"], "video_key")
    pred_by_video = group_by([row for row in pred_rows if row.get("clear") == "true"], "video_key")
    video_keys = sorted(set(gt_by_video) | set(pred_by_video))

    by_video = {
        video_key: score_video(gt_by_video.get(video_key, []), pred_by_video.get(video_key, []))
        for video_key in video_keys
    }
    macro = float(np.mean([item["final_score"] for item in by_video.values()])) if by_video else 0.0
    return {"macro_score": macro, "by_video": by_video}


def score_video(gt_rows: list[dict[str, str]], pred_rows: list[dict[str, str]]) -> dict[str, Any]:
    n_gt = len(gt_rows)
    n_pred = len(pred_rows)

    if n_gt == 0:
        fp_penalty = 1.0 if n_pred else 0.0
        final = max(0.0, 1.0 - fp_penalty)
        return {
            "final_score": final,
            "score_time": 1.0,
            "score_fighter": 1.0,
            "score_punch_type": 1.0,
            "score_effectiveness": 1.0,
            "score_hand": 1.0,
            "score_target": 1.0,
            "fp_penalty": fp_penalty,
            "n_gt": 0,
            "n_pred": n_pred,
            "n_tp": 0,
            "n_fp": n_pred,
            "n_fn": 0,
        }

    matches = match_events(gt_rows, pred_rows)
    n_tp = len(matches)
    n_fp = n_pred - n_tp
    n_fn = n_gt - n_tp

    score_time = _score_time(gt_rows, pred_rows, matches)
    score_fighter = _score_accuracy(gt_rows, pred_rows, matches, "fighter")
    score_hand = _score_accuracy(gt_rows, pred_rows, matches, "hand")
    score_target = _score_accuracy(gt_rows, pred_rows, matches, "target")
    score_punch_type = _score_balanced(gt_rows, pred_rows, matches, "punch_type")
    score_effectiveness = _score_balanced(
        gt_rows,
        pred_rows,
        matches,
        "effectiveness",
        value_map=EFFECTIVENESS_METRIC_MAP,
    )

    fp_penalty = n_fp / (n_gt + n_fp) if n_gt + n_fp else 0.0
    raw = (
        0.50 * score_time
        + 0.20 * score_fighter
        + 0.10 * score_punch_type
        + 0.08 * score_effectiveness
        + 0.06 * score_hand
        + 0.06 * score_target
        - fp_penalty
    )
    final = min(1.0, max(0.0, raw))

    return {
        "final_score": final,
        "score_time": score_time,
        "score_fighter": score_fighter,
        "score_punch_type": score_punch_type,
        "score_effectiveness": score_effectiveness,
        "score_hand": score_hand,
        "score_target": score_target,
        "fp_penalty": fp_penalty,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
    }


def match_events(gt_rows: list[dict[str, str]], pred_rows: list[dict[str, str]]) -> list[Match]:
    if not gt_rows or not pred_rows:
        return []

    gt_frames = np.array([as_int(row["frame"], "frame") for row in gt_rows], dtype=float)
    pred_frames = np.array([as_int(row["frame"], "frame") for row in pred_rows], dtype=float)
    frame_diffs = np.abs(gt_frames[:, None] - pred_frames[None, :])
    time_diffs = frame_diffs / FPS

    large = 1e9
    cost = np.full((len(gt_rows), len(pred_rows)), large, dtype=float)
    valid = time_diffs <= 1.0
    cost[valid] = time_diffs[valid]

    for gt_index, gt_row in enumerate(gt_rows):
        for pred_index, pred_row in enumerate(pred_rows):
            if valid[gt_index, pred_index] and gt_row["fighter"] == pred_row["fighter"]:
                cost[gt_index, pred_index] -= 1e-6

    gt_indices, pred_indices = linear_sum_assignment(cost)
    matches = [
        Match(int(gt_index), int(pred_index))
        for gt_index, pred_index in zip(gt_indices, pred_indices)
        if cost[gt_index, pred_index] < large / 2
    ]
    return matches


def _score_time(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    matches: list[Match],
) -> float:
    total = 0.0
    for match in matches:
        gt_frame = as_int(gt_rows[match.gt_index]["frame"], "frame")
        pred_frame = as_int(pred_rows[match.pred_index]["frame"], "frame")
        time_error = min(1.0, abs(gt_frame - pred_frame) / FPS / 0.5)
        total += 1.0 - time_error
    return total / len(gt_rows)


def _score_accuracy(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    matches: list[Match],
    column: str,
) -> float:
    correct = 0
    for match in matches:
        if _attributes_are_scorable(gt_rows, pred_rows, match) and (
            gt_rows[match.gt_index][column] == pred_rows[match.pred_index][column]
        ):
            correct += 1
    return correct / len(gt_rows)


def _score_balanced(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    matches: list[Match],
    column: str,
    value_map: dict[str, str] | None = None,
) -> float:
    def mapped(value: str) -> str:
        return value_map.get(value, value) if value_map else value

    gt_values = [mapped(row[column]) for row in gt_rows]
    counts = Counter(gt_values)
    denom = sum(1.0 / counts[value] for value in gt_values)
    if denom == 0:
        return 1.0

    total = 0.0
    for match in matches:
        if not _attributes_are_scorable(gt_rows, pred_rows, match):
            continue
        gt_value = mapped(gt_rows[match.gt_index][column])
        pred_value = mapped(pred_rows[match.pred_index][column])
        if gt_value == pred_value:
            total += 1.0 / counts[gt_value]
    return total / denom


def _attributes_are_scorable(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    match: Match,
) -> bool:
    gt_frame = as_int(gt_rows[match.gt_index]["frame"], "frame")
    pred_frame = as_int(pred_rows[match.pred_index]["frame"], "frame")
    return abs(gt_frame - pred_frame) / FPS / 0.5 < 1.0
