#!/usr/bin/env python3
"""Evaluate calibrated fixed-row fighter flips from dual, pose-rival, and appearance gates."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    score_pose_tracks,
)
from tools.evaluate_pose_selection_variants import video_wins


DEFAULT_BASE = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv"
)
DEFAULT_DUAL = Path(
    "data/processed/vit_features/"
    "dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv"
)
DEFAULT_APPEARANCE = Path(
    "data/processed/appearance_audits/"
    "dino_color_strict_clearens094012_train_yolo11s_20260522/train_fighter_appearance_video_summary.csv"
)
DEFAULT_OUTPUT_PREFIX = Path("data/processed/diagnostics/calibrated_fighter_flip_20260523")


@dataclass(frozen=True)
class PoseEvidence:
    self_score: float
    rival_score: float
    rival_frame: int | None
    rival_hand: str
    rival_target: str

    @property
    def diff(self) -> float:
        return self.rival_score - self.self_score

    @property
    def ratio(self) -> float:
        if self.self_score <= 1e-9:
            return self.rival_score / 0.05 if self.rival_score > 0.0 else 0.0
        return self.rival_score / self.self_score


@dataclass
class RowSignal:
    index: int
    row: dict[str, str]
    proposed_fighter: str
    dual_flip: bool
    appearance: dict[str, str]
    pose: dict[tuple[int, str], PoseEvidence]
    gt_fighter: str = ""
    gt_frame: str = ""
    gt_match_status: str = "unmatched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dual-candidate", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--tracks-dir", type=Path, default=Path("data/processed/pose_tracks/train_yolo11s_conf035"))
    parser.add_argument("--appearance-summary", type=Path, default=DEFAULT_APPEARANCE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--windows", default="0,2,4")
    parser.add_argument("--match-modes", default="same_hand_target,same_hand,any")
    parser.add_argument("--ratios", default="1.0,1.15,1.3,1.6")
    parser.add_argument("--min-rival-scores", default="0.2,0.5,0.8")
    parser.add_argument("--top-ns", default="5,8,10,12,15,20,25,30")
    parser.add_argument(
        "--rank-metrics",
        default="pose_diff,dual_first_pose_diff,appearance_pose_diff",
    )
    parser.add_argument(
        "--appearance-gates",
        default="all,calibration_ok,color_early_acc_ge_0.85,color_sep_ratio_ge_1.8",
    )
    parser.add_argument(
        "--sources",
        default="dual,dual_and_rival",
        help="Comma-separated sources: dual,dual_and_rival,dual_or_rival,rival.",
    )
    parser.add_argument("--max-flips", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--pass-fighter-delta", type=float, default=0.010)
    parser.add_argument("--pass-macro-delta", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_rows = [row for row in read_csv_rows(args.base) if row.get("clear") == "true"]
    dual_rows = [row for row in read_csv_rows(args.dual_candidate) if row.get("clear") == "true"]
    ensure_aligned(base_rows, dual_rows)

    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    keys = sorted({row["video_key"] for row in base_rows})
    missing = sorted(set(keys) - set(video_by_key))
    if missing:
        raise RuntimeError(f"Prediction keys are not train videos: {missing}")
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]

    baseline = score_predictions(gt_rows, base_rows)
    baseline_components = component_means(baseline)
    appearance_by_key = load_appearance(args.appearance_summary)
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    windows = parse_ints(args.windows)
    match_modes = parse_list(args.match_modes)
    candidates_by_key = {
        key: index_candidates(
            apply_temporal_context(score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config), config)
        )
        for key in keys
    }

    row_signals = build_row_signals(
        base_rows,
        dual_rows,
        gt_rows,
        appearance_by_key,
        candidates_by_key,
        windows,
        match_modes,
    )
    results = sweep_gates(args, row_signals, base_rows, gt_rows, videos, baseline, baseline_components)
    if not results:
        raise RuntimeError("No strict flip gates produced any changed rows")

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    write_summary(summary_path, results)

    best = max(results, key=lambda row: float(row["macro_score"]))
    best_indices = parse_index_set(str(best["selected_indices"]))
    best_rows = apply_flips(base_rows, row_signals, best_indices)
    rows_path = output_prefix.with_name(output_prefix.name + "_best_rows.csv")
    write_csv_rows(rows_path, best_rows, SUBMISSION_COLUMNS)

    flip_audit_path = output_prefix.with_name(output_prefix.name + "_best_flips.csv")
    write_flip_audit(flip_audit_path, row_signals, best_indices, best)

    root_path = output_prefix.with_name(output_prefix.name + "_best_root_deltas.csv")
    write_root_deltas(root_path, videos, baseline, score_predictions(gt_rows, best_rows))

    print_baseline(baseline, baseline_components, len(base_rows))
    print(f"dual_reference_changed={sum(item.dual_flip for item in row_signals)}", flush=True)
    print(f"wrote_summary={summary_path}", flush=True)
    print(f"wrote_best_rows={rows_path}", flush=True)
    print(f"wrote_best_flips={flip_audit_path}", flush=True)
    print(f"wrote_best_root_deltas={root_path}", flush=True)
    print_top(results, args.top_k)
    return 0


def ensure_aligned(base_rows: list[dict[str, str]], dual_rows: list[dict[str, str]]) -> None:
    if len(base_rows) != len(dual_rows):
        raise RuntimeError(f"base/dual row count mismatch: {len(base_rows)} != {len(dual_rows)}")
    keys = ["id", "video_key", "frame"]
    for index, (base, dual) in enumerate(zip(base_rows, dual_rows)):
        if any(base.get(key, "") != dual.get(key, "") for key in keys):
            raise RuntimeError(f"base/dual mismatch at row {index}: {base} vs {dual}")


def load_appearance(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        print(f"appearance_summary_missing={path}", flush=True)
        return {}
    return {row["video_key"]: row for row in read_csv_rows(path)}


def build_row_signals(
    base_rows: list[dict[str, str]],
    dual_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    appearance_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, dict[int, list[PunchCandidate]]],
    windows: list[int],
    match_modes: list[str],
) -> list[RowSignal]:
    gt_match_info = matched_gt_info(base_rows, gt_rows)
    signals = []
    for index, (row, dual) in enumerate(zip(base_rows, dual_rows)):
        proposed = opposite_fighter(row["fighter"])
        dual_flip = dual["fighter"] != row["fighter"]
        signal = RowSignal(
            index=index,
            row=row,
            proposed_fighter=proposed,
            dual_flip=dual_flip,
            appearance=appearance_by_key.get(row["video_key"], {}),
            pose={},
        )
        signal.gt_fighter, signal.gt_frame, signal.gt_match_status = gt_match_info.get(
            index,
            ("", "", "unmatched"),
        )
        for window in windows:
            for mode in match_modes:
                signal.pose[(window, mode)] = pose_evidence(
                    row,
                    proposed,
                    candidates_by_key[row["video_key"]],
                    window,
                    mode,
                )
        signals.append(signal)
    return signals


def matched_gt_info(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
) -> dict[int, tuple[str, str, str]]:
    gt_by_key = group_by(gt_rows, "video_key")
    pred_by_key: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        pred_by_key[row["video_key"]].append((index, row))

    output: dict[int, tuple[str, str, str]] = {}
    for key, indexed_rows in pred_by_key.items():
        local_rows = [row for _index, row in indexed_rows]
        for match in match_events(gt_by_key.get(key, []), local_rows):
            global_index = indexed_rows[match.pred_index][0]
            gt = gt_by_key[key][match.gt_index]
            output[global_index] = (gt["fighter"], gt["frame"], "matched")
    return output


def pose_evidence(
    row: dict[str, str],
    proposed_fighter: str,
    candidates_by_frame: dict[int, list[PunchCandidate]],
    window: int,
    match_mode: str,
) -> PoseEvidence:
    frame = as_int(row["frame"], "frame")
    self_candidate = best_candidate(
        candidates_by_frame,
        frame,
        window,
        pose_predicate(row, row["fighter"], match_mode),
    )
    rival = best_candidate(
        candidates_by_frame,
        frame,
        window,
        pose_predicate(row, proposed_fighter, match_mode),
    )
    return PoseEvidence(
        self_score=self_candidate.score if self_candidate else 0.0,
        rival_score=rival.score if rival else 0.0,
        rival_frame=rival.frame if rival else None,
        rival_hand=rival.hand if rival else "",
        rival_target=rival.target if rival else "",
    )


def pose_predicate(
    row: dict[str, str],
    fighter: str,
    match_mode: str,
) -> Callable[[PunchCandidate], bool]:
    if match_mode == "same_hand_target":
        return (
            lambda candidate: candidate.fighter == fighter
            and candidate.hand == row["hand"]
            and candidate.target == row["target"]
        )
    if match_mode == "same_hand":
        return lambda candidate: candidate.fighter == fighter and candidate.hand == row["hand"]
    if match_mode == "any":
        return lambda candidate: candidate.fighter == fighter
    raise ValueError(f"Unknown match mode: {match_mode}")


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


def index_candidates(candidates: list[PunchCandidate]) -> dict[int, list[PunchCandidate]]:
    by_frame: dict[int, list[PunchCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate.frame].append(candidate)
    return by_frame


def sweep_gates(
    args: argparse.Namespace,
    row_signals: list[RowSignal],
    base_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    videos: list[dict[str, str]],
    baseline: dict[str, Any],
    baseline_components: dict[str, float],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    seen_masks: set[tuple[int, ...]] = set()
    top_ns = [value for value in parse_ints(args.top_ns) if value <= args.max_flips]
    rank_metrics = parse_list(args.rank_metrics)
    appearance_gates = parse_list(args.appearance_gates)
    windows = parse_ints(args.windows)
    match_modes = parse_list(args.match_modes)

    sources = parse_list(args.sources)
    if "dual" in sources:
        for appearance_gate in appearance_gates:
            for window in windows:
                for match_mode in match_modes:
                    for rank_metric in rank_metrics:
                        for top_n in top_ns:
                            selected = select_rows(
                                row_signals,
                                "dual",
                                window,
                                match_mode,
                                0.0,
                                0.0,
                                rank_metric,
                                appearance_gate,
                                top_n,
                            )
                            maybe_score_result(
                                results,
                                seen_masks,
                                args,
                                base_rows,
                                gt_rows,
                                videos,
                                baseline,
                                baseline_components,
                                selected,
                                "dual",
                                window,
                                match_mode,
                                0.0,
                                0.0,
                                rank_metric,
                                appearance_gate,
                                top_n,
                            )

    gated_sources = [source for source in sources if source != "dual"]
    for source in gated_sources:
        for appearance_gate in appearance_gates:
            for window in windows:
                for match_mode in match_modes:
                    for ratio in parse_floats(args.ratios):
                        for min_rival_score in parse_floats(args.min_rival_scores):
                            for rank_metric in rank_metrics:
                                for top_n in top_ns:
                                    selected = select_rows(
                                        row_signals,
                                        source,
                                        window,
                                        match_mode,
                                        ratio,
                                        min_rival_score,
                                        rank_metric,
                                        appearance_gate,
                                        top_n,
                                    )
                                    maybe_score_result(
                                        results,
                                        seen_masks,
                                        args,
                                        base_rows,
                                        gt_rows,
                                        videos,
                                        baseline,
                                        baseline_components,
                                        selected,
                                        source,
                                        window,
                                        match_mode,
                                        ratio,
                                        min_rival_score,
                                        rank_metric,
                                        appearance_gate,
                                        top_n,
                                    )
    return results


def select_rows(
    row_signals: list[RowSignal],
    source: str,
    window: int,
    match_mode: str,
    ratio: float,
    min_rival_score: float,
    rank_metric: str,
    appearance_gate: str,
    top_n: int,
) -> list[RowSignal]:
    eligible = []
    for signal in row_signals:
        evidence = signal.pose[(window, match_mode)]
        rival_ok = evidence.rival_score >= min_rival_score and evidence.rival_score >= evidence.self_score * ratio
        if not appearance_ok(signal.appearance, appearance_gate):
            continue
        if source == "dual":
            ok = signal.dual_flip
        elif source == "dual_and_rival":
            ok = signal.dual_flip and rival_ok
        elif source == "dual_or_rival":
            ok = signal.dual_flip or rival_ok
        elif source == "rival":
            ok = rival_ok
        else:
            raise ValueError(f"Unknown source: {source}")
        if ok:
            eligible.append(signal)
    return sorted(
        eligible,
        key=lambda signal: rank_key(signal, signal.pose[(window, match_mode)], rank_metric),
        reverse=True,
    )[:top_n]


def maybe_score_result(
    results: list[dict[str, object]],
    seen_masks: set[tuple[int, ...]],
    args: argparse.Namespace,
    base_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    videos: list[dict[str, str]],
    baseline: dict[str, Any],
    baseline_components: dict[str, float],
    selected: list[RowSignal],
    source: str,
    window: int,
    match_mode: str,
    ratio: float,
    min_rival_score: float,
    rank_metric: str,
    appearance_gate: str,
    top_n: int,
) -> None:
    if not selected or len(selected) > args.max_flips:
        return
    mask = tuple(sorted(signal.index for signal in selected))
    if mask in seen_masks:
        return
    seen_masks.add(mask)
    rows = apply_flips(base_rows, selected, set(mask))
    score = score_predictions(gt_rows, rows)
    components = component_means(score)
    video_delta = changed_video_counts(selected)
    root_delta = root_counts(selected, videos)
    root_summary, negative_tournament_roots = root_delta_summary(videos, baseline, score)
    delta_macro = float(score["macro_score"]) - float(baseline["macro_score"])
    delta_fighter = components["fighter"] - baseline_components["fighter"]
    delta_time = components["time"] - baseline_components["time"]
    delta_fp = components["fp_penalty"] - baseline_components["fp_penalty"]
    pass_gate = (
        (delta_fighter >= args.pass_fighter_delta or delta_macro >= args.pass_macro_delta)
        and abs(delta_time) <= 1e-9
        and abs(delta_fp) <= 1e-9
        and not negative_tournament_roots
    )
    results.append(
        {
            "name": gate_name(source, window, match_mode, ratio, min_rival_score, rank_metric, appearance_gate, top_n),
            "macro_score": f"{float(score['macro_score']):.6f}",
            "delta_macro": f"{delta_macro:+.6f}",
            "score_fighter": f"{components['fighter']:.6f}",
            "delta_fighter": f"{delta_fighter:+.6f}",
            "time": f"{components['time']:.6f}",
            "delta_time": f"{delta_time:+.6f}",
            "fp_penalty": f"{components['fp_penalty']:.6f}",
            "delta_fp": f"{delta_fp:+.6f}",
            "wins": video_wins(score, baseline),
            "n_flips": len(selected),
            "source": source,
            "window": window,
            "match_mode": match_mode,
            "ratio": f"{ratio:g}",
            "min_rival_score": f"{min_rival_score:g}",
            "rank_metric": rank_metric,
            "appearance_gate": appearance_gate,
            "top_n": top_n,
            "flips_by_video": format_counts(video_delta),
            "flips_by_root": format_counts(root_delta),
            "root_deltas": root_summary,
            "negative_tournament_roots": ";".join(negative_tournament_roots),
            "pass_gate": str(pass_gate).lower(),
            "selected_indices": ";".join(str(index) for index in mask),
        }
    )


def apply_flips(
    base_rows: list[dict[str, str]],
    selected: list[RowSignal] | set[int],
    selected_indices: set[int] | None = None,
) -> list[dict[str, str]]:
    if selected_indices is None:
        if isinstance(selected, set):
            selected_indices = selected
        else:
            selected_indices = {signal.index for signal in selected}
    proposed_by_index = {}
    if not isinstance(selected, set):
        proposed_by_index = {signal.index: signal.proposed_fighter for signal in selected}
    output = []
    for index, row in enumerate(base_rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if index in selected_indices:
            item["fighter"] = proposed_by_index.get(index, opposite_fighter(row["fighter"]))
        output.append(item)
    return output


def rank_key(signal: RowSignal, evidence: PoseEvidence, rank_metric: str) -> tuple[float, ...]:
    app_score = appearance_score(signal.appearance)
    dual = 1.0 if signal.dual_flip else 0.0
    if rank_metric == "pose_diff":
        return (evidence.diff, evidence.rival_score, evidence.ratio, dual, app_score, -signal.index)
    if rank_metric == "pose_ratio":
        return (evidence.ratio, evidence.diff, evidence.rival_score, dual, app_score, -signal.index)
    if rank_metric == "rival_score":
        return (evidence.rival_score, evidence.diff, evidence.ratio, dual, app_score, -signal.index)
    if rank_metric == "dual_first_pose_diff":
        return (dual, evidence.diff, evidence.rival_score, evidence.ratio, app_score, -signal.index)
    if rank_metric == "appearance_pose_diff":
        return (app_score, evidence.diff, evidence.rival_score, evidence.ratio, dual, -signal.index)
    raise ValueError(f"Unknown rank metric: {rank_metric}")


def appearance_ok(row: dict[str, str], gate: str) -> bool:
    if gate == "all":
        return True
    if gate == "calibration_ok":
        return truthy(row.get("calibration_ok", ""))
    if "_ge_" in gate:
        column, threshold_text = gate.rsplit("_ge_", 1)
        return truthy(row.get("calibration_ok", "true")) and parse_optional_float(row.get(column, "")) >= float(
            threshold_text
        )
    raise ValueError(f"Unknown appearance gate: {gate}")


def appearance_score(row: dict[str, str]) -> float:
    if not row:
        return 0.0
    values = [
        parse_optional_float(row.get("color_early_acc", "")),
        parse_optional_float(row.get("color_nn_acc", "")),
        parse_optional_float(row.get("color_sep_ratio", "")) / 3.0,
    ]
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return 0.0
    return float(np.mean(finite)) + (0.25 if truthy(row.get("calibration_ok", "")) else 0.0)


def root_delta_summary(
    videos: list[dict[str, str]],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[str, list[str]]:
    video_by_key = {row["video_key"]: row for row in videos}
    roots: dict[str, list[str]] = defaultdict(list)
    for key in baseline["by_video"]:
        video = video_by_key[key]
        roots[root_label(video)].append(key)
    parts = []
    negative_tournament_roots = []
    for label, keys in sorted(roots.items()):
        base_value = aggregate_final(baseline, keys)
        candidate_value = aggregate_final(candidate, keys)
        delta = candidate_value - base_value
        parts.append(f"{label}:{delta:+.6f}")
        dataset_type = label.split("|", 1)[0]
        if dataset_type.startswith("tournament") and delta < -1e-9:
            negative_tournament_roots.append(f"{label}:{delta:+.6f}")
    return ";".join(parts), negative_tournament_roots


def write_root_deltas(
    path: Path,
    videos: list[dict[str, str]],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    video_by_key = {row["video_key"]: row for row in videos}
    roots: dict[str, list[str]] = defaultdict(list)
    for key in baseline["by_video"]:
        roots[root_label(video_by_key[key])].append(key)
    rows = []
    for label, keys in sorted(roots.items()):
        base_values = component_subset(baseline, keys)
        candidate_values = component_subset(candidate, keys)
        rows.append(
            {
                "root": label,
                "n_videos": len(keys),
                "base_macro": f"{base_values['macro']:.6f}",
                "candidate_macro": f"{candidate_values['macro']:.6f}",
                "delta_macro": f"{candidate_values['macro'] - base_values['macro']:+.6f}",
                "base_fighter": f"{base_values['fighter']:.6f}",
                "candidate_fighter": f"{candidate_values['fighter']:.6f}",
                "delta_fighter": f"{candidate_values['fighter'] - base_values['fighter']:+.6f}",
                "base_time": f"{base_values['time']:.6f}",
                "candidate_time": f"{candidate_values['time']:.6f}",
                "delta_time": f"{candidate_values['time'] - base_values['time']:+.6f}",
                "base_fp": f"{base_values['fp_penalty']:.6f}",
                "candidate_fp": f"{candidate_values['fp_penalty']:.6f}",
                "delta_fp": f"{candidate_values['fp_penalty'] - base_values['fp_penalty']:+.6f}",
                "videos": ";".join(keys),
            }
        )
    write_csv_rows(
        path,
        rows,
        [
            "root",
            "n_videos",
            "base_macro",
            "candidate_macro",
            "delta_macro",
            "base_fighter",
            "candidate_fighter",
            "delta_fighter",
            "base_time",
            "candidate_time",
            "delta_time",
            "base_fp",
            "candidate_fp",
            "delta_fp",
            "videos",
        ],
    )


def write_flip_audit(
    path: Path,
    row_signals: list[RowSignal],
    selected_indices: set[int],
    best: dict[str, object],
) -> None:
    columns = [
        "id",
        "video_key",
        "frame",
        "fighter_before",
        "fighter_after",
        "hand",
        "target",
        "punch_type",
        "effectiveness",
        "dual_flip",
        "gt_match_status",
        "gt_frame",
        "gt_fighter",
        "fighter_outcome",
        "source",
        "window",
        "match_mode",
        "self_score",
        "rival_score",
        "rival_ratio",
        "rival_diff",
        "rival_frame",
        "rival_hand",
        "rival_target",
        "calibration_ok",
        "color_early_acc",
        "color_nn_acc",
        "color_sep_ratio",
        "appearance_gate",
        "rank_metric",
    ]
    rows = []
    window = int(best["window"])
    match_mode = str(best["match_mode"])
    for index in sorted(selected_indices):
        signal = row_signals[index]
        row = signal.row
        evidence = signal.pose[(window, match_mode)]
        rows.append(
            {
                "id": row["id"],
                "video_key": row["video_key"],
                "frame": row["frame"],
                "fighter_before": row["fighter"],
                "fighter_after": signal.proposed_fighter,
                "hand": row["hand"],
                "target": row["target"],
                "punch_type": row["punch_type"],
                "effectiveness": row["effectiveness"],
                "dual_flip": str(signal.dual_flip).lower(),
                "gt_match_status": signal.gt_match_status,
                "gt_frame": signal.gt_frame,
                "gt_fighter": signal.gt_fighter,
                "fighter_outcome": fighter_outcome(signal),
                "source": best["source"],
                "window": window,
                "match_mode": match_mode,
                "self_score": f"{evidence.self_score:.6f}",
                "rival_score": f"{evidence.rival_score:.6f}",
                "rival_ratio": f"{evidence.ratio:.6f}",
                "rival_diff": f"{evidence.diff:.6f}",
                "rival_frame": "" if evidence.rival_frame is None else evidence.rival_frame,
                "rival_hand": evidence.rival_hand,
                "rival_target": evidence.rival_target,
                "calibration_ok": signal.appearance.get("calibration_ok", ""),
                "color_early_acc": signal.appearance.get("color_early_acc", ""),
                "color_nn_acc": signal.appearance.get("color_nn_acc", ""),
                "color_sep_ratio": signal.appearance.get("color_sep_ratio", ""),
                "appearance_gate": best["appearance_gate"],
                "rank_metric": best["rank_metric"],
            }
        )
    write_csv_rows(path, rows, columns)


def fighter_outcome(signal: RowSignal) -> str:
    if not signal.gt_fighter:
        return "unmatched"
    before_correct = signal.row["fighter"] == signal.gt_fighter
    after_correct = signal.proposed_fighter == signal.gt_fighter
    if not before_correct and after_correct:
        return "fix"
    if before_correct and not after_correct:
        return "break"
    if before_correct and after_correct:
        return "unchanged_correct"
    return "still_wrong"


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "name",
        "macro_score",
        "delta_macro",
        "score_fighter",
        "delta_fighter",
        "time",
        "delta_time",
        "fp_penalty",
        "delta_fp",
        "wins",
        "n_flips",
        "source",
        "window",
        "match_mode",
        "ratio",
        "min_rival_score",
        "rank_metric",
        "appearance_gate",
        "top_n",
        "flips_by_video",
        "flips_by_root",
        "root_deltas",
        "negative_tournament_roots",
        "pass_gate",
        "selected_indices",
    ]
    sorted_rows = sorted(rows, key=lambda row: float(row["macro_score"]), reverse=True)
    write_csv_rows(path, sorted_rows, columns)


def print_baseline(score: dict[str, Any], components: dict[str, float], n_rows: int) -> None:
    print(
        f"baseline macro={float(score['macro_score']):.6f} "
        f"fighter={components['fighter']:.6f} time={components['time']:.6f} "
        f"fp={components['fp_penalty']:.6f} n={n_rows}",
        flush=True,
    )


def print_top(rows: list[dict[str, object]], limit: int) -> None:
    columns = [
        "macro_score",
        "delta_macro",
        "score_fighter",
        "delta_fighter",
        "time",
        "fp_penalty",
        "wins",
        "n_flips",
        "source",
        "window",
        "match_mode",
        "ratio",
        "min_rival_score",
        "rank_metric",
        "appearance_gate",
        "top_n",
        "pass_gate",
        "flips_by_video",
        "negative_tournament_roots",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: float(item["macro_score"]), reverse=True)[:limit]:
        writer.writerow({column: row[column] for column in columns})


def component_means(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "macro": float(score["macro_score"]),
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "punch_type": float(np.mean([item["score_punch_type"] for item in by_video.values()])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def component_subset(score: dict[str, Any], keys: list[str]) -> dict[str, float]:
    values = [score["by_video"][key] for key in keys]
    return {
        "macro": float(np.mean([item["final_score"] for item in values])),
        "fighter": float(np.mean([item["score_fighter"] for item in values])),
        "time": float(np.mean([item["score_time"] for item in values])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in values])),
    }


def aggregate_final(score: dict[str, Any], keys: list[str]) -> float:
    return float(np.mean([score["by_video"][key]["final_score"] for key in keys]))


def changed_video_counts(selected: list[RowSignal]) -> Counter[str]:
    return Counter(signal.row["video_key"] for signal in selected)


def root_counts(selected: list[RowSignal], videos: list[dict[str, str]]) -> Counter[str]:
    video_by_key = {row["video_key"]: row for row in videos}
    return Counter(root_label(video_by_key[signal.row["video_key"]]) for signal in selected)


def root_label(video: dict[str, str]) -> str:
    return f"{video.get('dataset_type', '')}|{video.get('data_root', '')}"


def gate_name(
    source: str,
    window: int,
    match_mode: str,
    ratio: float,
    min_rival_score: float,
    rank_metric: str,
    appearance_gate: str,
    top_n: int,
) -> str:
    return (
        f"{source}_w{window}_{match_mode}_r{ratio:g}_min{min_rival_score:g}_"
        f"{rank_metric}_{appearance_gate}_top{top_n}"
    )


def format_counts(counts: Counter[str]) -> str:
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def parse_index_set(text: str) -> set[int]:
    return {int(value) for value in text.split(";") if value != ""}


def opposite_fighter(value: str) -> str:
    if value == "red":
        return "blue"
    if value == "blue":
        return "red"
    raise ValueError(f"Unknown fighter label: {value}")


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_optional_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
