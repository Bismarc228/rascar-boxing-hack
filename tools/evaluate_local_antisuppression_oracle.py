#!/usr/bin/env python3
"""Oracle audit for local anti-suppression replacements.

This is not a submission generator. It measures whether the proposed local
anti-suppression shape has validation headroom before training a deployable
selector.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions, score_video


DEFAULT_PREDICTIONS = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_"
    "posehgb094_vitlogreg012_rows_20260522.csv"
)
DEFAULT_CASES = Path("data/processed/timing_audits/strict_clearens_yolo26l_pool_cases_20260523.csv")
DEFAULT_PAIRS = Path("data/processed/timing_audits/strict_clearens_yolo26l_pool_suppressor_pairs_20260523.csv")
COMPONENT_FIELDS = {
    "time": "score_time",
    "fighter": "score_fighter",
    "punch_type": "score_punch_type",
    "effectiveness": "score_effectiveness",
    "hand": "score_hand",
    "target": "score_target",
    "fp_penalty": "fp_penalty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--pool-cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--suppressor-pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--pair-gaps", default="2,4,10")
    parser.add_argument("--replace-windows", default="10,30,90")
    parser.add_argument("--suppressor-types", default="all,same_group,cross_group")
    parser.add_argument("--max-proposals-per-gt", type=int, default=3)
    parser.add_argument("--max-replacements-per-video", type=int, default=20)
    parser.add_argument("--same-nms-frames", type=int, default=10)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--require-no-nms-conflict", action="store_true")
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--output-selected", type=Path)
    parser.add_argument("--write-best-rows", type=Path)
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
    baseline_summary = score_summary(baseline)
    cases = read_csv_rows(args.pool_cases)
    pairs = read_csv_rows(args.suppressor_pairs)
    proposals = build_proposals(cases, pairs)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")

    summary_rows = []
    selected_by_config: dict[str, list[dict[str, object]]] = {}
    rows_by_config: dict[str, list[dict[str, str]]] = {}
    for pair_gap in parse_ints(args.pair_gaps):
        for replace_window in parse_ints(args.replace_windows):
            for suppressor_type in parse_types(args.suppressor_types):
                selected_rows, candidate_rows, proposal_count = evaluate_config(
                    proposals,
                    pred_by_key,
                    gt_by_key,
                    pair_gap,
                    replace_window,
                    suppressor_type,
                    args,
                )
                score = score_predictions(gt_rows, candidate_rows)
                summary = score_summary(score)
                config_id = config_name(pair_gap, replace_window, suppressor_type, args.require_no_nms_conflict)
                selected_by_config[config_id] = selected_rows
                rows_by_config[config_id] = candidate_rows
                row: dict[str, object] = {
                    "config": config_id,
                    "pair_gap": pair_gap,
                    "replace_window": replace_window,
                    "suppressor_type": suppressor_type,
                    "require_no_nms_conflict": str(args.require_no_nms_conflict).lower(),
                    "proposal_count": proposal_count,
                    "accepted": len(selected_rows),
                    "macro_score": f"{float(score['macro_score']):.6f}",
                    "macro_delta": f"{float(score['macro_score']) - float(baseline['macro_score']):+.6f}",
                }
                for name, value in summary.items():
                    row[name] = f"{value:.6f}"
                    row[f"{name}_delta"] = f"{value - baseline_summary[name]:+.6f}"
                summary_rows.append(row)

    summary_rows.sort(key=lambda row: float(row["macro_score"]), reverse=True)
    print(
        f"baseline macro={float(baseline['macro_score']):.6f} "
        f"time={baseline_summary['time']:.6f} fighter={baseline_summary['fighter']:.6f} "
        f"fp_penalty={baseline_summary['fp_penalty']:.6f} rows={len(pred_rows)}"
    )
    print("config,pair_gap,replace_window,suppressor_type,accepted,macro_delta,time_delta,fighter_delta,fp_penalty_delta")
    for row in summary_rows[: args.top_k]:
        print(
            ",".join(
                str(row[column])
                for column in [
                    "config",
                    "pair_gap",
                    "replace_window",
                    "suppressor_type",
                    "accepted",
                    "macro_delta",
                    "time_delta",
                    "fighter_delta",
                    "fp_penalty_delta",
                ]
            )
        )

    if args.output_summary:
        write_csv_rows(args.output_summary, summary_rows, summary_columns())
        print(f"wrote_summary={args.output_summary}")
    if args.output_selected and summary_rows:
        best_config = str(summary_rows[0]["config"])
        write_csv_rows(args.output_selected, selected_by_config[best_config], selected_columns())
        print(f"wrote_selected={args.output_selected} config={best_config}")
    if args.write_best_rows and summary_rows:
        best_config = str(summary_rows[0]["config"])
        write_csv_rows(args.write_best_rows, rows_by_config[best_config], SUBMISSION_COLUMNS)
        print(f"wrote_best_rows={args.write_best_rows} config={best_config}")
    return 0


def build_proposals(cases: list[dict[str, str]], pairs: list[dict[str, str]]) -> list[dict[str, object]]:
    case_by_key = {gt_key(row): row for row in cases if row.get("case_type") == "pool_covered_not_selected_scorable"}
    output = []
    for pair in pairs:
        if pair.get("has_suppressor") != "true":
            continue
        case = case_by_key.get(gt_key(pair))
        if case is None:
            continue
        positive_frame = as_int(pair["positive_frame"], "positive_frame")
        suppressor_frame = as_int(pair["suppressor_frame"], "suppressor_frame")
        nearest_selected_frame = optional_int(case.get("nearest_selected_frame", ""))
        output.append(
            {
                "video_key": pair["video_key"],
                "gt_frame": as_int(pair["gt_frame"], "gt_frame"),
                "gt_fighter": pair["gt_fighter"],
                "positive_frame": positive_frame,
                "positive_fighter": pair["positive_fighter"],
                "positive_hand": pair["positive_hand"],
                "positive_target": pair["positive_target"],
                "positive_score": float(pair["positive_score"]),
                "positive_rank": as_int(pair["positive_rank"], "positive_rank"),
                "suppressor_frame": suppressor_frame,
                "suppressor_fighter": pair["suppressor_fighter"],
                "suppressor_hand": pair["suppressor_hand"],
                "suppressor_target": pair["suppressor_target"],
                "suppressor_score": float(pair["suppressor_score"]),
                "suppressor_rank": as_int(pair["suppressor_rank"], "suppressor_rank"),
                "suppressor_type": pair["suppressor_type"],
                "score_ratio": float(pair["score_ratio"]),
                "pair_gap": abs(positive_frame - suppressor_frame),
                "nearest_selected_frame": nearest_selected_frame,
            }
        )
    return output


def evaluate_config(
    proposals: list[dict[str, object]],
    pred_by_key: dict[str, list[dict[str, str]]],
    gt_by_key: dict[str, list[dict[str, str]]],
    pair_gap: int,
    replace_window: int,
    suppressor_type: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, str]], int]:
    selected_audit = []
    output_rows = []
    proposal_count = 0
    proposals_by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    for proposal in proposals:
        if int(proposal["pair_gap"]) > pair_gap:
            continue
        if suppressor_type != "all" and proposal["suppressor_type"] != suppressor_type:
            continue
        proposals_by_key[str(proposal["video_key"])].append(proposal)
        proposal_count += 1

    for key in sorted(pred_by_key):
        current = [dict(row) for row in pred_by_key[key]]
        baseline_score = score_video(gt_by_key.get(key, []), current)
        used_gt: set[tuple[str, int]] = set()
        used_old_ids: set[str] = set()
        accepted_for_video = 0
        ordered = sorted(
            proposals_by_key.get(key, []),
            key=lambda item: (
                int(item["pair_gap"]),
                abs(int(item["positive_frame"]) - int(item["gt_frame"])),
                float(item["score_ratio"]),
            ),
        )
        for proposal in ordered:
            if accepted_for_video >= args.max_replacements_per_video:
                break
            gt_id = (str(proposal["video_key"]), int(proposal["gt_frame"]))
            if gt_id in used_gt:
                continue
            best = best_replacement(
                current,
                gt_by_key.get(key, []),
                proposal,
                replace_window,
                used_old_ids,
                args,
            )
            if best is None:
                continue
            old_index, row_after, score_after, delta = best
            old_row = current[old_index]
            current[old_index] = row_after
            used_old_ids.add(old_row["id"])
            used_gt.add(gt_id)
            accepted_for_video += 1
            selected_audit.append(
                {
                    "video_key": key,
                    "old_id": old_row["id"],
                    "old_frame": old_row["frame"],
                    "new_frame": row_after["frame"],
                    "gt_frame": proposal["gt_frame"],
                    "old_fighter": old_row["fighter"],
                    "new_fighter": row_after["fighter"],
                    "old_hand": old_row["hand"],
                    "new_hand": row_after["hand"],
                    "old_target": old_row["target"],
                    "new_target": row_after["target"],
                    "suppressor_type": proposal["suppressor_type"],
                    "pair_gap": proposal["pair_gap"],
                    "score_ratio": f"{float(proposal['score_ratio']):.6f}",
                    "delta_score": f"{delta:.9f}",
                    "video_score_after": f"{float(score_after['final_score']):.9f}",
                    "video_score_before_config": f"{float(baseline_score['final_score']):.9f}",
                }
            )
        output_rows.extend(current)
    output_rows.sort(key=lambda row: as_int(row["id"], "id"))
    return selected_audit, output_rows, proposal_count


def best_replacement(
    rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    proposal: dict[str, object],
    replace_window: int,
    used_old_ids: set[str],
    args: argparse.Namespace,
) -> tuple[int, dict[str, str], dict[str, object], float] | None:
    current_score = score_video(gt_rows, rows)
    positive_frame = int(proposal["positive_frame"])
    suppressor_frame = int(proposal["suppressor_frame"])
    preferred_center = int(proposal.get("nearest_selected_frame") or suppressor_frame)
    candidates = []
    for index, row in enumerate(rows):
        if row["id"] in used_old_ids:
            continue
        old_frame = as_int(row["frame"], "frame")
        if min(abs(old_frame - positive_frame), abs(old_frame - suppressor_frame), abs(old_frame - preferred_center)) > replace_window:
            continue
        candidates.append((index, row))
    candidates.sort(
        key=lambda item: (
            abs(as_int(item[1]["frame"], "frame") - suppressor_frame),
            abs(as_int(item[1]["frame"], "frame") - positive_frame),
        )
    )
    candidates = candidates[: args.max_proposals_per_gt]

    best: tuple[int, dict[str, str], dict[str, object], float] | None = None
    for index, old_row in candidates:
        row_after = make_replacement_row(old_row, proposal)
        if same_event(old_row, row_after):
            continue
        if args.require_no_nms_conflict and creates_nms_conflict(
            row_after,
            rows,
            index,
            args.same_nms_frames,
            args.cross_nms_frames,
        ):
            continue
        trial = [dict(row) for row in rows]
        trial[index] = row_after
        score_after = score_video(gt_rows, trial)
        delta = float(score_after["final_score"]) - float(current_score["final_score"])
        if delta <= 1e-12:
            continue
        if best is None or delta > best[3]:
            best = (index, row_after, score_after, delta)
    return best


def make_replacement_row(old_row: dict[str, str], proposal: dict[str, object]) -> dict[str, str]:
    row = {column: old_row.get(column, "") for column in SUBMISSION_COLUMNS}
    row["frame"] = str(proposal["positive_frame"])
    row["fighter"] = str(proposal["positive_fighter"])
    row["hand"] = str(proposal["positive_hand"])
    row["target"] = str(proposal["positive_target"])
    row["clear"] = "true"
    return row


def creates_nms_conflict(
    candidate: dict[str, str],
    rows: list[dict[str, str]],
    skip_index: int,
    same_nms: int,
    cross_nms: int,
) -> bool:
    candidate_frame = as_int(candidate["frame"], "frame")
    candidate_group = (candidate["fighter"], candidate["hand"])
    for index, row in enumerate(rows):
        if index == skip_index:
            continue
        frame_distance = abs(candidate_frame - as_int(row["frame"], "frame"))
        group = (row["fighter"], row["hand"])
        if candidate_group == group and frame_distance <= same_nms:
            return True
        if candidate_group != group and frame_distance <= cross_nms:
            return True
    return False


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        name: float(np.mean([item[field] for item in by_video.values()]))
        for name, field in COMPONENT_FIELDS.items()
    }


def gt_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row["video_key"],
        row["gt_frame"],
        row["gt_fighter"],
        row["gt_hand"],
        row["gt_target"],
        row.get("gt_punch_type", ""),
        row.get("gt_effectiveness", ""),
    )


def same_event(left: dict[str, str], right: dict[str, str]) -> bool:
    return (
        left["frame"] == right["frame"]
        and left["fighter"] == right["fighter"]
        and left["hand"] == right["hand"]
        and left["target"] == right["target"]
    )


def config_name(pair_gap: int, replace_window: int, suppressor_type: str, no_conflict: bool) -> str:
    suffix = "noconflict" if no_conflict else "metricguard"
    return f"gap{pair_gap}_w{replace_window}_{suppressor_type}_{suffix}"


def summary_columns() -> list[str]:
    columns = [
        "config",
        "pair_gap",
        "replace_window",
        "suppressor_type",
        "require_no_nms_conflict",
        "proposal_count",
        "accepted",
        "macro_score",
        "macro_delta",
    ]
    for name in COMPONENT_FIELDS:
        columns.extend([name, f"{name}_delta"])
    return columns


def selected_columns() -> list[str]:
    return [
        "video_key",
        "old_id",
        "old_frame",
        "new_frame",
        "gt_frame",
        "old_fighter",
        "new_fighter",
        "old_hand",
        "new_hand",
        "old_target",
        "new_target",
        "suppressor_type",
        "pair_gap",
        "score_ratio",
        "delta_score",
        "video_score_after",
        "video_score_before_config",
    ]


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_types(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def optional_int(text: str) -> int | None:
    return int(text) if text else None


if __name__ == "__main__":
    raise SystemExit(main())
