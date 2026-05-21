#!/usr/bin/env python3
"""Evaluate fixed-row fighter flips using nearby opposite-fighter pose candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    score_pose_tracks,
)
from tools.evaluate_pose_selection_variants import video_wins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--windows", default="0,2,4,8")
    parser.add_argument("--ratios", default="0.8,1.0,1.15,1.3,1.6,2.0")
    parser.add_argument("--min-rival-scores", default="0.0,0.2,0.5,0.8,1.0")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
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
    candidates_by_key = {
        key: index_candidates(apply_temporal_context(score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config), config))
        for key in keys
    }

    results = []
    for window in parse_ints(args.windows):
        for match_mode in ["same_hand_target", "same_hand", "any"]:
            predicate_factory = rival_predicate_factory(match_mode)
            for ratio in parse_floats(args.ratios):
                for min_rival_score in parse_floats(args.min_rival_scores):
                    for update_mode in ["fighter_only", "fighter_hand_target"]:
                        rows, changed = apply_rival_rule(
                            pred_rows,
                            candidates_by_key,
                            window,
                            predicate_factory,
                            ratio,
                            min_rival_score,
                            update_mode,
                        )
                        if changed == 0:
                            continue
                        score = score_predictions(gt_rows, rows)
                        summary = score_summary(score)
                        results.append(
                            {
                                "score": score["macro_score"],
                                "delta": score["macro_score"] - baseline["macro_score"],
                                "fighter": summary["fighter"],
                                "time": summary["time"],
                                "hand": summary["hand"],
                                "target": summary["target"],
                                "fp": summary["fp"],
                                "wins": video_wins(score, baseline),
                                "changed": changed,
                                "window": window,
                                "match_mode": match_mode,
                                "ratio": ratio,
                                "min_rival_score": min_rival_score,
                                "update_mode": update_mode,
                            }
                        )

    print(
        "score,delta,fighter,time,hand,target,fp,wins,n_changed,"
        "window,match_mode,ratio,min_rival_score,update_mode"
    )
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print(
            f"{item['score']:.6f},{item['delta']:.6f},{item['fighter']:.6f},"
            f"{item['time']:.6f},{item['hand']:.6f},{item['target']:.6f},"
            f"{item['fp']:.6f},{item['wins']},{item['changed']},"
            f"{item['window']},{item['match_mode']},{item['ratio']},"
            f"{item['min_rival_score']},{item['update_mode']}"
        )
    print_rival_diagnostics(pred_rows, gt_rows, candidates_by_key)
    return 0


def index_candidates(candidates: list[PunchCandidate]) -> dict[int, list[PunchCandidate]]:
    by_frame: dict[int, list[PunchCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate.frame].append(candidate)
    return by_frame


def apply_rival_rule(
    rows: list[dict[str, str]],
    candidates_by_key: dict[str, dict[int, list[PunchCandidate]]],
    window: int,
    predicate_factory: Callable[[dict[str, str]], Callable[[PunchCandidate], bool]],
    ratio: float,
    min_rival_score: float,
    update_mode: str,
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for row in rows:
        new_row = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        frame = as_int(row["frame"], "frame")
        candidates = candidates_by_key[row["video_key"]]
        self_candidate = best_candidate(
            candidates,
            frame,
            window,
            lambda candidate, row=row: candidate.fighter == row["fighter"]
            and candidate.hand == row["hand"]
            and candidate.target == row["target"],
        )
        if self_candidate is None:
            self_candidate = best_candidate(
                candidates,
                frame,
                window,
                lambda candidate, row=row: candidate.fighter == row["fighter"],
            )
        rival = best_candidate(candidates, frame, window, predicate_factory(row))
        self_score = self_candidate.score if self_candidate else 0.0
        if rival is None or rival.score < min_rival_score:
            output.append(new_row)
            continue
        if rival.score < self_score * ratio:
            output.append(new_row)
            continue
        if update_mode == "fighter_only":
            new_row["fighter"] = rival.fighter
        elif update_mode == "fighter_hand_target":
            new_row["fighter"] = rival.fighter
            new_row["hand"] = rival.hand
            new_row["target"] = rival.target
        else:
            raise ValueError(f"Unknown update_mode: {update_mode}")
        if any(new_row[col] != row.get(col, "") for col in ["fighter", "hand", "target"]):
            changed += 1
        output.append(new_row)
    return output, changed


def rival_predicate_factory(
    match_mode: str,
) -> Callable[[dict[str, str]], Callable[[PunchCandidate], bool]]:
    if match_mode == "same_hand_target":
        return lambda row: (
            lambda candidate: candidate.fighter != row["fighter"]
            and candidate.hand == row["hand"]
            and candidate.target == row["target"]
        )
    if match_mode == "same_hand":
        return lambda row: (
            lambda candidate: candidate.fighter != row["fighter"] and candidate.hand == row["hand"]
        )
    if match_mode == "any":
        return lambda row: (lambda candidate: candidate.fighter != row["fighter"])
    raise ValueError(f"Unknown match_mode: {match_mode}")


def best_candidate(
    candidates_by_frame: dict[int, list[PunchCandidate]],
    frame: int,
    window: int,
    predicate: Callable[[PunchCandidate], bool],
) -> PunchCandidate | None:
    best: PunchCandidate | None = None
    best_key = (-1.0, 0.0)
    for current_frame in range(frame - window, frame + window + 1):
        for candidate in candidates_by_frame.get(current_frame, []):
            if not predicate(candidate):
                continue
            rank = (candidate.score, -abs(candidate.frame - frame))
            if rank > best_key:
                best = candidate
                best_key = rank
    return best


def print_rival_diagnostics(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    candidates_by_key: dict[str, dict[int, list[PunchCandidate]]],
) -> None:
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    matched_total = 0
    wrong_total = 0
    rival_would_fix = Counter()
    rival_would_break = Counter()
    from rascar_boxing.metric import match_events

    for key, rows in pred_by_key.items():
        gt_video = gt_by_key.get(key, [])
        for match in match_events(gt_video, rows):
            pred = rows[match.pred_index]
            gt = gt_video[match.gt_index]
            frame = as_int(pred["frame"], "frame")
            candidates = candidates_by_key[key]
            matched_total += 1
            is_wrong = pred["fighter"] != gt["fighter"]
            wrong_total += int(is_wrong)
            for window in [0, 2, 4, 8]:
                rival = best_candidate(
                    candidates,
                    frame,
                    window,
                    lambda candidate, pred=pred: candidate.fighter != pred["fighter"],
                )
                if rival is None:
                    continue
                label = f"window_{window}"
                if is_wrong and rival.fighter == gt["fighter"]:
                    rival_would_fix[label] += 1
                if not is_wrong and rival.fighter != gt["fighter"]:
                    rival_would_break[label] += 1
    print("diagnostic,total_matches,wrong_matches,signal,would_fix,would_break")
    for label in sorted(set(rival_would_fix) | set(rival_would_break)):
        print(
            f"rival,{matched_total},{wrong_total},{label},"
            f"{rival_would_fix[label]},{rival_would_break[label]}"
        )


def print_score(label: str, score: dict[str, object], wins: int, n_rows: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},fighter={summary['fighter']:.6f},"
        f"time={summary['time']:.6f},fp={summary['fp']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
        "fp": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
