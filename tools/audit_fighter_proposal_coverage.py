#!/usr/bin/env python3
"""Audit fixed-row fighter-flip proposal coverage before training a gate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_fighter_rival_flip import (
    best_candidate,
    index_candidates,
    rival_predicate_factory,
)


DEFAULT_BASE = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_"
    "posehgb094_vitlogreg012_rows_20260522.csv"
)
DEFAULT_DUAL = Path(
    "data/processed/vit_features/"
    "dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv"
)
COMPONENTS = {
    "time": "score_time",
    "fighter": "score_fighter",
    "fp_penalty": "fp_penalty",
}


@dataclass(frozen=True)
class MatchLabel:
    matched: bool
    scorable: bool
    gt_fighter: str
    base_wrong: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dual-candidate", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--tracks-dir", type=Path, action="append", default=[])
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--windows", default="0,2,4,8")
    parser.add_argument("--match-modes", default="same_hand_target,same_hand,any")
    parser.add_argument("--ratios", default="0.8,1.0,1.15,1.3,1.6,2.0")
    parser.add_argument("--min-rival-scores", default="0.0,0.2,0.5,0.8,1.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_rows = [row for row in read_csv_rows(args.base) if row.get("clear") == "true"]
    if not base_rows:
        raise SystemExit(f"no clear rows in {args.base}")
    keys = sorted({row["video_key"] for row in base_rows})
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    labels = build_match_labels(base_rows, gt_rows)
    wrong_total = sum(label.matched and label.base_wrong for label in labels)
    baseline = score_predictions(gt_rows, base_rows)
    baseline_components = component_means(baseline)

    result_rows: list[dict[str, object]] = []
    if args.dual_candidate and args.dual_candidate.exists():
        dual_rows = [row for row in read_csv_rows(args.dual_candidate) if row.get("clear") == "true"]
        ensure_aligned(base_rows, dual_rows)
        proposals = [
            dual["fighter"] if dual["fighter"] != base["fighter"] else ""
            for base, dual in zip(base_rows, dual_rows)
        ]
        result_rows.append(
            evaluate_proposals(
                "dual",
                "dual_candidate",
                base_rows,
                gt_rows,
                labels,
                wrong_total,
                baseline,
                baseline_components,
                proposals,
                {},
            )
        )

    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    for tracks_dir in args.tracks_dir:
        candidates_by_key = {
            key: index_candidates(
                apply_temporal_context(score_pose_tracks(tracks_dir / f"{key}.jsonl", config), config)
            )
            for key in keys
        }
        for window in parse_ints(args.windows):
            for match_mode in split_values(args.match_modes):
                predicate_factory = rival_predicate_factory(match_mode)
                for ratio in parse_floats(args.ratios):
                    for min_rival_score in parse_floats(args.min_rival_scores):
                        proposals = build_rival_proposals(
                            base_rows,
                            candidates_by_key,
                            window,
                            predicate_factory,
                            ratio,
                            min_rival_score,
                        )
                        if not any(proposals):
                            continue
                        result_rows.append(
                            evaluate_proposals(
                                "rival",
                                tracks_dir.name,
                                base_rows,
                                gt_rows,
                                labels,
                                wrong_total,
                                baseline,
                                baseline_components,
                                proposals,
                                {
                                    "window": window,
                                    "match_mode": match_mode,
                                    "ratio": f"{ratio:g}",
                                    "min_rival_score": f"{min_rival_score:g}",
                                },
                            )
                        )

    result_rows.sort(
        key=lambda row: (
            float(row["oracle_fighter_delta"]),
            int(row["fix"]),
            -int(row["break"]),
            float(row["all_fighter_delta"]),
        ),
        reverse=True,
    )
    write_csv_rows(args.output, result_rows, output_columns())
    print_baseline(baseline, baseline_components, len(base_rows), wrong_total)
    print(f"wrote={args.output} rows={len(result_rows)}")
    print_top(result_rows, args.top_k)
    return 0


def build_match_labels(rows: list[dict[str, str]], gt_rows: list[dict[str, str]]) -> list[MatchLabel]:
    labels = [MatchLabel(False, False, "", False) for _ in rows]
    for match in match_events(gt_rows, rows):
        row = rows[match.pred_index]
        gt = gt_rows[match.gt_index]
        scorable = abs(as_int(gt["frame"], "frame") - as_int(row["frame"], "frame")) / FPS / 0.5 < 1.0
        labels[match.pred_index] = MatchLabel(
            True,
            scorable,
            gt["fighter"],
            scorable and row["fighter"] != gt["fighter"],
        )
    return labels


def ensure_aligned(base_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]) -> None:
    if len(base_rows) != len(candidate_rows):
        raise RuntimeError(f"row count mismatch: {len(base_rows)} != {len(candidate_rows)}")
    keys = ["id", "video_key", "frame", "punch_type", "hand", "target", "effectiveness"]
    for index, (base, candidate) in enumerate(zip(base_rows, candidate_rows)):
        if any(base.get(key, "") != candidate.get(key, "") for key in keys):
            raise RuntimeError(f"row mismatch at {index}: {base} vs {candidate}")


def build_rival_proposals(
    rows: list[dict[str, str]],
    candidates_by_key: dict[str, dict[int, list[object]]],
    window: int,
    predicate_factory,
    ratio: float,
    min_rival_score: float,
) -> list[str]:
    proposals = []
    for row in rows:
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
        self_score = float(self_candidate.score) if self_candidate else 0.0
        if rival is None or rival.score < min_rival_score or rival.score < self_score * ratio:
            proposals.append("")
        else:
            proposals.append(rival.fighter)
    return proposals


def evaluate_proposals(
    source: str,
    source_name: str,
    base_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    labels: list[MatchLabel],
    wrong_total: int,
    baseline: dict[str, object],
    baseline_components: dict[str, float],
    proposals: list[str],
    params: dict[str, object],
) -> dict[str, object]:
    selected_all = {index for index, proposal in enumerate(proposals) if proposal}
    fix_indices = set()
    counts = Counter()
    for index, proposal in enumerate(proposals):
        if not proposal:
            continue
        label = labels[index]
        if not label.matched:
            counts["unmatched"] += 1
        elif not label.scorable:
            counts["time_only"] += 1
        elif label.base_wrong and proposal == label.gt_fighter:
            counts["fix"] += 1
            fix_indices.add(index)
        elif not label.base_wrong and proposal != label.gt_fighter:
            counts["break"] += 1
        elif label.base_wrong:
            counts["wrong_not_fixed"] += 1
        else:
            counts["neutral"] += 1

    all_score = score_predictions(gt_rows, apply_flips(base_rows, proposals, selected_all))
    oracle_score = score_predictions(gt_rows, apply_flips(base_rows, proposals, fix_indices))
    all_components = component_means(all_score)
    oracle_components = component_means(oracle_score)
    row = {
        "source": source,
        "source_name": source_name,
        "window": params.get("window", ""),
        "match_mode": params.get("match_mode", ""),
        "ratio": params.get("ratio", ""),
        "min_rival_score": params.get("min_rival_score", ""),
        "proposed": len(selected_all),
        "fix": counts["fix"],
        "break": counts["break"],
        "unmatched": counts["unmatched"],
        "time_only": counts["time_only"],
        "wrong_not_fixed": counts["wrong_not_fixed"],
        "neutral": counts["neutral"],
        "wrong_total": wrong_total,
        "wrong_coverage": f"{ratio_value(counts['fix'], wrong_total):.6f}",
        "oracle_selected": len(fix_indices),
        "oracle_macro_delta": f"{float(oracle_score['macro_score']) - float(baseline['macro_score']):+.6f}",
        "oracle_fighter_delta": f"{oracle_components['fighter'] - baseline_components['fighter']:+.6f}",
        "all_macro_delta": f"{float(all_score['macro_score']) - float(baseline['macro_score']):+.6f}",
        "all_fighter_delta": f"{all_components['fighter'] - baseline_components['fighter']:+.6f}",
        "all_time_delta": f"{all_components['time'] - baseline_components['time']:+.6f}",
        "all_fp_penalty_delta": f"{all_components['fp_penalty'] - baseline_components['fp_penalty']:+.6f}",
        "all_min_video_delta": f"{min_video_delta(all_score, baseline):+.6f}",
    }
    return row


def apply_flips(
    rows: list[dict[str, str]],
    proposals: list[str],
    selected: set[int],
) -> list[dict[str, str]]:
    output = []
    for index, row in enumerate(rows):
        item = {column: row.get(column, "") for column in SUBMISSION_COLUMNS}
        if index in selected and proposals[index]:
            item["fighter"] = proposals[index]
        output.append(item)
    return output


def component_means(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        name: float(np.mean([item[field] for item in by_video.values()]))
        for name, field in COMPONENTS.items()
    }


def min_video_delta(candidate: dict[str, object], baseline: dict[str, object]) -> float:
    values = []
    for key, base_item in baseline["by_video"].items():
        item = candidate["by_video"].get(key)
        final = float(item["final_score"]) if item is not None else 0.0
        values.append(final - float(base_item["final_score"]))
    return min(values) if values else 0.0


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in split_values(value)]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in split_values(value)]


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ratio_value(numerator: int, denominator: int) -> float:
    return float(numerator) / max(1.0, float(denominator))


def output_columns() -> list[str]:
    return [
        "source",
        "source_name",
        "window",
        "match_mode",
        "ratio",
        "min_rival_score",
        "proposed",
        "fix",
        "break",
        "unmatched",
        "time_only",
        "wrong_not_fixed",
        "neutral",
        "wrong_total",
        "wrong_coverage",
        "oracle_selected",
        "oracle_macro_delta",
        "oracle_fighter_delta",
        "all_macro_delta",
        "all_fighter_delta",
        "all_time_delta",
        "all_fp_penalty_delta",
        "all_min_video_delta",
    ]


def print_baseline(
    score: dict[str, object],
    components: dict[str, float],
    n_rows: int,
    wrong_total: int,
) -> None:
    print(
        f"baseline macro={float(score['macro_score']):.6f} "
        f"fighter={components['fighter']:.6f} time={components['time']:.6f} "
        f"fp_penalty={components['fp_penalty']:.6f} rows={n_rows} "
        f"matched_wrong_fighter={wrong_total}",
        flush=True,
    )


def print_top(rows: list[dict[str, object]], limit: int) -> None:
    print(
        "source,source_name,window,match_mode,ratio,min_rival_score,proposed,fix,break,"
        "unmatched,oracle_fighter_delta,oracle_macro_delta,all_fighter_delta,all_macro_delta,"
        "all_time_delta,all_fp_penalty_delta,all_min_video_delta"
    )
    for row in rows[:limit]:
        print(
            ",".join(
                str(row[column])
                for column in [
                    "source",
                    "source_name",
                    "window",
                    "match_mode",
                    "ratio",
                    "min_rival_score",
                    "proposed",
                    "fix",
                    "break",
                    "unmatched",
                    "oracle_fighter_delta",
                    "oracle_macro_delta",
                    "all_fighter_delta",
                    "all_macro_delta",
                    "all_time_delta",
                    "all_fp_penalty_delta",
                    "all_min_video_delta",
                ]
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
