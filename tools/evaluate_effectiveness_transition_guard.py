#!/usr/bin/env python3
"""Evaluate fixed-row effectiveness transition guards against a strict anchor."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import EFFECTIVENESS_METRIC_MAP, FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions


FIXED_COLUMNS = ["clear", "frame", "fighter", "hand", "target", "punch_type"]
EPS = 1e-12


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class VariantSpec:
    name: str
    from_values: frozenset[str] | None
    to_values: frozenset[str] | None


VARIANTS = [
    VariantSpec("landed_to_miss", frozenset({"landed"}), frozenset({"miss"})),
    VariantSpec("landed_to_blocked", frozenset({"landed"}), frozenset({"blocked"})),
    VariantSpec("landed_to_nonlanded", frozenset({"landed"}), frozenset({"miss", "blocked"})),
    VariantSpec("blocked_changes", frozenset({"blocked"}), frozenset({"landed", "miss"})),
    VariantSpec("miss_changes", frozenset({"miss"}), frozenset({"landed", "blocked"})),
    VariantSpec("nonlanded_changes", frozenset({"blocked", "miss"}), frozenset({"landed", "blocked", "miss"})),
    VariantSpec("blocked_miss_swap", frozenset({"blocked", "miss"}), frozenset({"blocked", "miss"})),
    VariantSpec("miss_to_landed_control", frozenset({"miss"}), frozenset({"landed"})),
    VariantSpec("blocked_to_landed_control", frozenset({"blocked"}), frozenset({"landed"})),
    VariantSpec("all_effectiveness_changes", None, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate prediction as name=path. Repeat for multiple candidates.",
    )
    parser.add_argument(
        "--key-columns",
        default="id,video_key",
        help="Comma-separated columns used to align candidate rows to anchor rows.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or 'all'.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("data/processed/diagnostics/effectiveness_transition_guard_20260523"),
    )
    parser.add_argument(
        "--rows-output-dir",
        type=Path,
        default=Path("data/processed/vit_features/effectiveness_transition_guard_rows_20260523"),
    )
    parser.add_argument("--write-row-artifacts", action="store_true")
    parser.add_argument(
        "--row-artifact-variants",
        default="landed_to_miss,landed_to_blocked,landed_to_nonlanded,nonlanded_changes,miss_to_landed_control,all_effectiveness_changes",
    )
    parser.add_argument("--pass-effectiveness-delta", type=float, default=0.020)
    parser.add_argument("--pass-weighted-macro-delta", type=float, default=0.0015)
    parser.add_argument("--pass-min-wins", type=int, default=8)
    parser.add_argument("--pass-max-bad-effectiveness-videos", type=int, default=2)
    parser.add_argument("--bad-effectiveness-delta", type=float, default=-0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = [parse_candidate_spec(value) for value in args.candidate]
    variants = select_variants(args.variants)
    row_artifact_variants = set(split_csv(args.row_artifact_variants))
    key_columns = split_csv(args.key_columns)

    anchor_rows = read_csv_rows(args.anchor)
    if not anchor_rows:
        raise SystemExit(f"empty anchor: {args.anchor}")
    validate_anchor_rows(anchor_rows)
    video_keys = sorted({row["video_key"] for row in anchor_rows if row.get("clear") == "true"})
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(video_keys)
    ]
    baseline_score = score_predictions(gt_rows, clear_rows(anchor_rows))
    baseline_components = component_means(baseline_score)
    gt_match_by_index = matched_gt_by_anchor_index(gt_rows, anchor_rows)

    summary_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for candidate in candidates:
        candidate_rows = read_csv_rows(candidate.path)
        proposal_by_key, audit = build_proposal_index(candidate_rows, anchor_rows, key_columns)
        audit_rows.append(
            {
                "candidate": candidate.name,
                "path": str(candidate.path),
                **audit,
            }
        )
        for variant in variants:
            guarded_rows, changed_indices = apply_guard(anchor_rows, proposal_by_key, key_columns, variant)
            verify_fixed_columns(anchor_rows, guarded_rows)
            score = score_predictions(gt_rows, clear_rows(guarded_rows))
            components = component_means(score)
            changes = summarize_changes(anchor_rows, guarded_rows, changed_indices)
            video_rows = collect_per_video_rows(
                candidate.name,
                variant.name,
                baseline_score,
                score,
                changes.by_video,
                args.bad_effectiveness_delta,
            )
            per_video_rows.extend(video_rows)
            wins = sum(float(row["delta_weighted_macro"]) > EPS for row in video_rows)
            losses = sum(float(row["delta_weighted_macro"]) < -EPS for row in video_rows)
            ties = len(video_rows) - wins - losses
            bad_eff_videos = sum(
                float(row["delta_effectiveness"]) < args.bad_effectiveness_delta for row in video_rows
            )
            weighted_macro_delta = float(score["macro_score"]) - float(baseline_score["macro_score"])
            effectiveness_delta = components["effectiveness"] - baseline_components["effectiveness"]
            pass_effectiveness = effectiveness_delta >= args.pass_effectiveness_delta
            pass_weighted_macro = weighted_macro_delta >= args.pass_weighted_macro_delta
            pass_wins = wins >= args.pass_min_wins
            pass_risk = bad_eff_videos <= args.pass_max_bad_effectiveness_videos
            summary_rows.append(
                {
                    "candidate": candidate.name,
                    "variant": variant.name,
                    "n_rows": len(anchor_rows),
                    "n_clear_rows": len(clear_rows(anchor_rows)),
                    "candidate_rows": len(candidate_rows),
                    "matched_anchor_rows": audit["matched_anchor_rows"],
                    "missing_anchor_rows": audit["missing_anchor_rows"],
                    "extra_candidate_rows": audit["extra_candidate_rows"],
                    "fixed_mismatch_rows": audit["fixed_mismatch_rows"],
                    "changed_rows": len(changed_indices),
                    "transition_counts": format_counts(changes.transitions),
                    "weighted_macro": f"{float(score['macro_score']):.6f}",
                    "weighted_macro_delta": f"{weighted_macro_delta:+.6f}",
                    "score_effectiveness": f"{components['effectiveness']:.6f}",
                    "score_effectiveness_delta": f"{effectiveness_delta:+.6f}",
                    "score_time_delta": f"{components['time'] - baseline_components['time']:+.6f}",
                    "score_fighter_delta": f"{components['fighter'] - baseline_components['fighter']:+.6f}",
                    "score_punch_type_delta": f"{components['punch_type'] - baseline_components['punch_type']:+.6f}",
                    "score_hand_delta": f"{components['hand'] - baseline_components['hand']:+.6f}",
                    "score_target_delta": f"{components['target'] - baseline_components['target']:+.6f}",
                    "fp_penalty_delta": f"{components['fp_penalty'] - baseline_components['fp_penalty']:+.6f}",
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "bad_effectiveness_videos": bad_eff_videos,
                    "pass_effectiveness": str(pass_effectiveness).lower(),
                    "pass_weighted_macro": str(pass_weighted_macro).lower(),
                    "pass_wins": str(pass_wins).lower(),
                    "pass_risk": str(pass_risk).lower(),
                    "pass_all": str(pass_effectiveness and pass_weighted_macro and pass_wins and pass_risk).lower(),
                }
            )
            confusion_rows.extend(
                collect_transition_confusion(
                    candidate.name,
                    variant.name,
                    anchor_rows,
                    guarded_rows,
                    changed_indices,
                    gt_match_by_index,
                )
            )
            if args.write_row_artifacts and variant.name in row_artifact_variants:
                out_path = args.rows_output_dir / f"{safe_name(candidate.name)}_{variant.name}_rows.csv"
                write_csv_rows(out_path, guarded_rows, SUBMISSION_COLUMNS)

    write_outputs(args.output_prefix, summary_rows, per_video_rows, confusion_rows, audit_rows)
    print_console_report(args.anchor, baseline_score, baseline_components, summary_rows, args.output_prefix)
    return 0


@dataclass
class ChangeSummary:
    transitions: Counter[str]
    by_video: dict[str, dict[str, object]]


def parse_candidate_spec(value: str) -> CandidateSpec:
    if "=" not in value:
        raise ValueError(f"--candidate must be name=path, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"candidate name is empty in {value!r}")
    return CandidateSpec(name, Path(path))


def select_variants(text: str) -> list[VariantSpec]:
    if text == "all":
        return VARIANTS
    names = set(split_csv(text))
    by_name = {variant.name: variant for variant in VARIANTS}
    unknown = sorted(names - set(by_name))
    if unknown:
        raise ValueError(f"unknown variants: {','.join(unknown)}")
    return [variant for variant in VARIANTS if variant.name in names]


def split_csv(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def validate_anchor_rows(rows: list[dict[str, str]]) -> None:
    missing = [column for column in SUBMISSION_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"anchor missing columns: {','.join(missing)}")
    non_clear = sum(row.get("clear") != "true" for row in rows)
    if non_clear:
        raise ValueError(f"anchor contains non-clear rows: {non_clear}")


def clear_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("clear") == "true"]


def make_key(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in columns)


def build_proposal_index(
    candidate_rows: list[dict[str, str]],
    anchor_rows: list[dict[str, str]],
    key_columns: list[str],
) -> tuple[dict[tuple[str, ...], dict[str, str]], dict[str, int]]:
    proposal_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    duplicates = 0
    for row in candidate_rows:
        key = make_key(row, key_columns)
        if key in proposal_by_key:
            duplicates += 1
            continue
        proposal_by_key[key] = row

    anchor_keys = [make_key(row, key_columns) for row in anchor_rows]
    anchor_key_set = set(anchor_keys)
    matched = 0
    fixed_mismatch_rows = 0
    for anchor_row, key in zip(anchor_rows, anchor_keys):
        candidate_row = proposal_by_key.get(key)
        if candidate_row is None:
            continue
        matched += 1
        if any(anchor_row.get(column, "") != candidate_row.get(column, "") for column in FIXED_COLUMNS):
            fixed_mismatch_rows += 1
    extra = sum(1 for key in proposal_by_key if key not in anchor_key_set)
    return proposal_by_key, {
        "matched_anchor_rows": matched,
        "missing_anchor_rows": len(anchor_rows) - matched,
        "extra_candidate_rows": extra,
        "duplicate_candidate_keys": duplicates,
        "fixed_mismatch_rows": fixed_mismatch_rows,
    }


def apply_guard(
    anchor_rows: list[dict[str, str]],
    proposal_by_key: dict[tuple[str, ...], dict[str, str]],
    key_columns: list[str],
    variant: VariantSpec,
) -> tuple[list[dict[str, str]], list[int]]:
    output: list[dict[str, str]] = []
    changed_indices: list[int] = []
    for index, anchor in enumerate(anchor_rows):
        item = {column: anchor.get(column, "") for column in SUBMISSION_COLUMNS}
        proposal = proposal_by_key.get(make_key(anchor, key_columns))
        proposed_effectiveness = proposal.get("effectiveness", anchor["effectiveness"]) if proposal else anchor["effectiveness"]
        if should_accept_transition(anchor["effectiveness"], proposed_effectiveness, variant):
            item["effectiveness"] = proposed_effectiveness
            changed_indices.append(index)
        output.append(item)
    return output, changed_indices


def should_accept_transition(current: str, proposed: str, variant: VariantSpec) -> bool:
    if proposed == current:
        return False
    if variant.from_values is not None and current not in variant.from_values:
        return False
    if variant.to_values is not None and proposed not in variant.to_values:
        return False
    return True


def verify_fixed_columns(anchor_rows: list[dict[str, str]], guarded_rows: list[dict[str, str]]) -> None:
    for index, (anchor, guarded) in enumerate(zip(anchor_rows, guarded_rows)):
        for column in FIXED_COLUMNS:
            if anchor.get(column, "") != guarded.get(column, ""):
                raise AssertionError(f"fixed column changed at row {index}: {column}")


def component_means(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    if not isinstance(by_video, dict) or not by_video:
        raise RuntimeError("score result has no by_video entries")
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "punch_type": float(np.mean([item["score_punch_type"] for item in by_video.values()])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def summarize_changes(
    anchor_rows: list[dict[str, str]],
    guarded_rows: list[dict[str, str]],
    changed_indices: list[int],
) -> ChangeSummary:
    transitions: Counter[str] = Counter()
    by_video: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "changed_rows": 0,
            "transitions": Counter(),
        }
    )
    for index in changed_indices:
        anchor = anchor_rows[index]
        guarded = guarded_rows[index]
        transition = f"{anchor['effectiveness']}->{guarded['effectiveness']}"
        transitions[transition] += 1
        video_item = by_video[anchor["video_key"]]
        video_item["changed_rows"] = int(video_item["changed_rows"]) + 1
        video_item["transitions"][transition] += 1
    return ChangeSummary(transitions=transitions, by_video=by_video)


def collect_per_video_rows(
    candidate: str,
    variant: str,
    baseline_score: dict[str, object],
    candidate_score: dict[str, object],
    changes_by_video: dict[str, dict[str, object]],
    bad_effectiveness_delta: float,
) -> list[dict[str, object]]:
    rows = []
    base_by_video = baseline_score["by_video"]
    candidate_by_video = candidate_score["by_video"]
    for video_key in sorted(base_by_video):
        base = base_by_video[video_key]
        item = candidate_by_video[video_key]
        change_item = changes_by_video.get(video_key, {})
        transitions = change_item.get("transitions", Counter())
        delta_score = float(item["final_score"]) - float(base["final_score"])
        delta_eff = float(item["score_effectiveness"]) - float(base["score_effectiveness"])
        rows.append(
            {
                "candidate": candidate,
                "variant": variant,
                "video_key": video_key,
                "changed_rows": int(change_item.get("changed_rows", 0)),
                "transition_counts": format_counts(transitions if isinstance(transitions, Counter) else Counter()),
                "base_weighted_macro": f"{float(base['final_score']):.6f}",
                "candidate_weighted_macro": f"{float(item['final_score']):.6f}",
                "delta_weighted_macro": f"{delta_score:+.6f}",
                "base_effectiveness": f"{float(base['score_effectiveness']):.6f}",
                "candidate_effectiveness": f"{float(item['score_effectiveness']):.6f}",
                "delta_effectiveness": f"{delta_eff:+.6f}",
                "bad_effectiveness_video": str(delta_eff < bad_effectiveness_delta).lower(),
            }
        )
    return rows


def matched_gt_by_anchor_index(
    gt_rows: list[dict[str, str]],
    anchor_rows: list[dict[str, str]],
) -> dict[int, tuple[dict[str, str], bool]]:
    gt_by_key = group_by(gt_rows, "video_key")
    anchor_by_key = group_by(anchor_rows, "video_key")
    global_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(anchor_rows):
        global_indices[row["video_key"]].append(index)
    output: dict[int, tuple[dict[str, str], bool]] = {}
    for video_key, rows in anchor_by_key.items():
        for match in match_events(gt_by_key.get(video_key, []), rows):
            gt = gt_by_key[video_key][match.gt_index]
            pred = rows[match.pred_index]
            global_index = global_indices[video_key][match.pred_index]
            output[global_index] = (gt, is_scorable(gt, pred))
    return output


def is_scorable(gt_row: dict[str, str], pred_row: dict[str, str]) -> bool:
    gt_frame = as_int(gt_row["frame"], "frame")
    pred_frame = as_int(pred_row["frame"], "frame")
    return abs(gt_frame - pred_frame) / FPS / 0.5 < 1.0


def collect_transition_confusion(
    candidate: str,
    variant: str,
    anchor_rows: list[dict[str, str]],
    guarded_rows: list[dict[str, str]],
    changed_indices: list[int],
    gt_match_by_index: dict[int, tuple[dict[str, str], bool]],
) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for index in changed_indices:
        anchor = anchor_rows[index]
        guarded = guarded_rows[index]
        transition = f"{anchor['effectiveness']}->{guarded['effectiveness']}"
        gt_pair = gt_match_by_index.get(index)
        if gt_pair is None:
            gt_effectiveness = "__unmatched__"
            scorable = False
        else:
            gt, scorable = gt_pair
            gt_effectiveness = gt["effectiveness"] if scorable else "__unscored__"
        key = (transition, gt_effectiveness)
        counts[key]["count"] += 1
        if gt_effectiveness.startswith("__"):
            counts[key]["unscored_or_unmatched"] += 1
            continue
        base_correct = metric_effectiveness(anchor["effectiveness"]) == metric_effectiveness(gt_effectiveness)
        candidate_correct = metric_effectiveness(guarded["effectiveness"]) == metric_effectiveness(gt_effectiveness)
        counts[key]["base_correct"] += int(base_correct)
        counts[key]["candidate_correct"] += int(candidate_correct)
        counts[key]["improved"] += int((not base_correct) and candidate_correct)
        counts[key]["worsened"] += int(base_correct and (not candidate_correct))
        counts[key]["same_correctness"] += int(base_correct == candidate_correct)

    rows = []
    for (transition, gt_effectiveness), counter in sorted(counts.items()):
        rows.append(
            {
                "candidate": candidate,
                "variant": variant,
                "transition": transition,
                "gt_effectiveness": gt_effectiveness,
                "count": counter["count"],
                "base_correct": counter["base_correct"],
                "candidate_correct": counter["candidate_correct"],
                "improved": counter["improved"],
                "worsened": counter["worsened"],
                "net_correct": counter["improved"] - counter["worsened"],
                "same_correctness": counter["same_correctness"],
                "unscored_or_unmatched": counter["unscored_or_unmatched"],
            }
        )
    return rows


def metric_effectiveness(value: str) -> str:
    return EFFECTIVENESS_METRIC_MAP.get(value, value)


def format_counts(counts: Counter[str] | dict[str, int]) -> str:
    if not counts:
        return ""
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def write_outputs(
    output_prefix: Path,
    summary_rows: list[dict[str, object]],
    per_video_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
) -> None:
    write_csv_rows(Path(str(output_prefix) + "_summary.csv"), summary_rows, list(summary_rows[0]))
    write_csv_rows(Path(str(output_prefix) + "_per_video.csv"), per_video_rows, list(per_video_rows[0]))
    if confusion_rows:
        write_csv_rows(Path(str(output_prefix) + "_transition_confusion.csv"), confusion_rows, list(confusion_rows[0]))
    else:
        write_csv_rows(
            Path(str(output_prefix) + "_transition_confusion.csv"),
            [],
            [
                "candidate",
                "variant",
                "transition",
                "gt_effectiveness",
                "count",
                "base_correct",
                "candidate_correct",
                "improved",
                "worsened",
                "net_correct",
                "same_correctness",
                "unscored_or_unmatched",
            ],
        )
    write_csv_rows(Path(str(output_prefix) + "_candidate_audit.csv"), audit_rows, list(audit_rows[0]))


def print_console_report(
    anchor: Path,
    baseline_score: dict[str, object],
    baseline_components: dict[str, float],
    summary_rows: list[dict[str, object]],
    output_prefix: Path,
) -> None:
    print(
        "baseline,"
        f"anchor={anchor},weighted_macro={float(baseline_score['macro_score']):.6f},"
        f"score_effectiveness={baseline_components['effectiveness']:.6f}"
    )
    print(
        "candidate,variant,changed_rows,weighted_macro_delta,"
        "score_effectiveness_delta,wins,losses,bad_effectiveness_videos,pass_all"
    )
    sorted_rows = sorted(
        summary_rows,
        key=lambda row: (
            float(row["weighted_macro_delta"]),
            float(row["score_effectiveness_delta"]),
        ),
        reverse=True,
    )
    for row in sorted_rows[:20]:
        print(
            f"{row['candidate']},{row['variant']},{row['changed_rows']},"
            f"{row['weighted_macro_delta']},{row['score_effectiveness_delta']},"
            f"{row['wins']},{row['losses']},{row['bad_effectiveness_videos']},{row['pass_all']}"
        )
    print(f"wrote_prefix={output_prefix}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
