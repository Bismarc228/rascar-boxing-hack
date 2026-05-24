#!/usr/bin/env python3
"""OOF audit for cluster-first local timing replacements."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions, score_video
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    score_pose_tracks,
)
from tools.audit_pose_pool_cluster_first import row_features
from tools.evaluate_exchange_state_gate import BASE_FEATURES, fight_group
from tools.evaluate_pose_selection_variants import video_wins


SWEEP_COLUMNS = [
    "threshold",
    "macro",
    "delta_macro",
    "time",
    "delta_time",
    "fighter",
    "delta_fighter",
    "fp",
    "delta_fp",
    "wins",
    "n_rows",
    "replaced",
    "oracle_hits",
    "oracle_misses",
    "pass_macro",
    "pass_time",
    "pass_fp",
    "pass_all",
]

SELECTED_COLUMNS = [
    "threshold",
    "rank",
    "video_key",
    "pred_index",
    "old_frame",
    "new_frame",
    "old_fighter",
    "new_fighter",
    "old_hand",
    "new_hand",
    "old_target",
    "new_target",
    "cluster_p",
    "candidate_p",
    "score",
    "is_oracle_replacement",
]


@dataclass(frozen=True)
class Proposal:
    video_key: str
    pred_index: int
    old_row: dict[str, str]
    candidate: PunchCandidate
    cluster_p: float
    cluster_rank: int
    global_rank: int
    local_distance_rank: int
    local_score_rank: int
    local_size: int
    row_feature_values: tuple[float, ...]
    candidate_row_feature_values: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--oracle-audit", type=Path, required=True)
    parser.add_argument("--cluster-first-probs", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--proposal-window", type=int, default=90)
    parser.add_argument("--rank-cap", type=int, default=10000)
    parser.add_argument("--max-first-stage-rows", type=int, default=160)
    parser.add_argument("--cluster-min-prob", type=float, default=0.0)
    parser.add_argument("--max-candidates-per-row", type=int, default=160)
    parser.add_argument("--max-replacements-total", type=int, default=40)
    parser.add_argument("--max-replacements-per-video", type=int, default=6)
    parser.add_argument("--target-mode", choices=["exact_oracle", "single_delta"], default="exact_oracle")
    parser.add_argument("--model", choices=["hgb", "rf", "et", "hgbreg", "rfreg", "etreg"], default="et")
    parser.add_argument("--combine-mode", choices=["candidate", "product", "sqrt_product"], default="product")
    parser.add_argument(
        "--thresholds",
        default="0.80,0.70,0.60,0.50,0.40,0.30,0.25,0.20,0.15,0.10,0.075,0.05,0.025",
    )
    parser.add_argument("--output-sweep", type=Path)
    parser.add_argument("--output-selected", type=Path)
    parser.add_argument("--output-rows", type=Path)
    parser.add_argument("--write-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    baseline_rows = [row for row in read_csv_rows(args.baseline_predictions) if row.get("clear") == "true"]
    pred_by_key = group_by(baseline_rows, "video_key")
    keys = sorted(pred_by_key)
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, baseline_rows)
    cluster_probs = read_cluster_probs(args.cluster_first_probs)
    accepted = accepted_keys(args.oracle_audit)
    accepted_rows = {(key, pred_index) for key, pred_index, *_ in accepted}
    cluster_rank = build_cluster_rank(cluster_probs)
    first_stage_rows = {
        key
        for key, prob in cluster_probs.items()
        if prob >= args.cluster_min_prob and cluster_rank[key] <= args.max_first_stage_rows
    }
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )

    proposals_by_key: dict[str, list[Proposal]] = {}
    for key in keys:
        proposals_by_key[key] = build_proposals(
            args,
            key,
            pred_by_key[key],
            video_by_key[key],
            config,
            cluster_probs,
            cluster_rank,
            first_stage_rows,
        )
    proposals = [proposal for key in keys for proposal in proposals_by_key[key]]
    y = np.asarray([1 if proposal_key(proposal) in accepted else 0 for proposal in proposals], dtype=np.int8)
    x = np.asarray([proposal_features(proposal, video_by_key[proposal.video_key]) for proposal in proposals], dtype=np.float32)
    groups = np.asarray([fight_group(video_by_key[proposal.video_key]) for proposal in proposals])
    if args.target_mode == "single_delta":
        target = single_replacement_deltas(keys, pred_by_key, group_by(gt_rows, "video_key"), proposals_by_key)
        y = (target > 1e-12).astype(np.int8)
    else:
        target = y.astype(np.float32)
    candidate_score = fit_oof_scores(args.model, x, y, target, groups)
    combined = combine_scores(args.combine_mode, proposals, candidate_score)

    first_stage_oracle_rows = len(first_stage_rows & accepted_rows)
    print_score("baseline", baseline, 0, len(baseline_rows), 0, 0, 0)
    print(
        f"first_stage_rows={len(first_stage_rows)} first_stage_oracle_rows={first_stage_oracle_rows}/"
        f"{len(accepted_rows)} proposals={len(proposals)} positive_labels={int(y.sum())} "
        f"oracle_replacements={len(accepted)} pos_rate={float(y.mean()) if len(y) else 0.0:.6f}"
    )
    print_candidate_metrics(y, candidate_score, combined, target)

    sweep_rows = []
    rows_by_threshold = {}
    selected_by_threshold = {}
    for threshold in parse_floats(args.thresholds):
        rows, selected = apply_replacements(
            args,
            threshold,
            keys,
            pred_by_key,
            proposals_by_key,
            candidate_score,
            combined,
            accepted,
        )
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        base_summary = score_summary(baseline)
        oracle_hits = sum(1 for proposal, *_ in selected if proposal_key(proposal) in accepted)
        oracle_misses = len(selected) - oracle_hits
        item = {
            "threshold": f"{threshold:.8g}",
            "macro": f"{float(score['macro_score']):.6f}",
            "delta_macro": f"{float(score['macro_score']) - float(baseline['macro_score']):+.6f}",
            "time": f"{summary['time']:.6f}",
            "delta_time": f"{summary['time'] - base_summary['time']:+.6f}",
            "fighter": f"{summary['fighter']:.6f}",
            "delta_fighter": f"{summary['fighter'] - base_summary['fighter']:+.6f}",
            "fp": f"{summary['fp']:.6f}",
            "delta_fp": f"{summary['fp'] - base_summary['fp']:+.6f}",
            "wins": str(video_wins(score, baseline)),
            "n_rows": str(len(rows)),
            "replaced": str(len(selected)),
            "oracle_hits": str(oracle_hits),
            "oracle_misses": str(oracle_misses),
            "pass_macro": str(float(score["macro_score"]) - float(baseline["macro_score"]) >= 0.005).lower(),
            "pass_time": str(summary["time"] - base_summary["time"] >= 0.008).lower(),
            "pass_fp": str(summary["fp"] <= base_summary["fp"] + 1e-12).lower(),
            "pass_all": str(
                float(score["macro_score"]) - float(baseline["macro_score"]) >= 0.005
                and summary["time"] - base_summary["time"] >= 0.008
                and summary["fp"] <= base_summary["fp"] + 1e-12
            ).lower(),
        }
        sweep_rows.append(item)
        rows_by_threshold[threshold] = rows
        selected_by_threshold[threshold] = selected
        print_sweep_row(item)

    best = max(sweep_rows, key=lambda row: float(row["macro"]))
    print(
        "best: "
        f"threshold={best['threshold']} macro={best['macro']} delta_macro={best['delta_macro']} "
        f"time={best['time']} delta_time={best['delta_time']} fp={best['fp']} "
        f"delta_fp={best['delta_fp']} replaced={best['replaced']} pass_all={best['pass_all']}"
    )

    if args.output_sweep:
        write_csv_rows(args.output_sweep, sweep_rows, SWEEP_COLUMNS)
    if args.output_selected:
        threshold = args.write_threshold if args.write_threshold is not None else float(best["threshold"])
        write_csv_rows(args.output_selected, selected_rows(threshold, selected_by_threshold[threshold], accepted), SELECTED_COLUMNS)
    if args.output_rows:
        threshold = args.write_threshold if args.write_threshold is not None else float(best["threshold"])
        write_csv_rows(args.output_rows, rows_by_threshold[threshold], SUBMISSION_COLUMNS)
    return 0


def read_cluster_probs(path: Path) -> dict[tuple[str, int], float]:
    output = {}
    for row in read_csv_rows(path):
        output[(row["video_key"], int(row["pred_index"]))] = float(row["p_replace"])
    return output


def accepted_keys(path: Path) -> set[tuple[str, int, int, str, str, str]]:
    output = set()
    for row in read_csv_rows(path):
        output.add(
            (
                row["video_key"],
                int(row["pred_index"]),
                int(row["new_frame"]),
                row["new_fighter"],
                row["new_hand"],
                row["new_target"],
            )
        )
    return output


def build_cluster_rank(cluster_probs: dict[tuple[str, int], float]) -> dict[tuple[str, int], int]:
    ranked = sorted(cluster_probs, key=lambda key: cluster_probs[key], reverse=True)
    return {key: index for index, key in enumerate(ranked, start=1)}


def build_proposals(
    args: argparse.Namespace,
    video_key: str,
    pred_rows: list[dict[str, str]],
    video: dict[str, str],
    config: PoseHeuristicConfig,
    cluster_probs: dict[tuple[str, int], float],
    cluster_rank: dict[tuple[str, int], int],
    first_stage_rows: set[tuple[str, int]],
) -> list[Proposal]:
    candidates = apply_temporal_context(
        score_pose_tracks(args.tracks_dir / f"{video_key}.jsonl", config),
        config,
    )
    global_ranks = {
        candidate_key(candidate): rank
        for rank, candidate in enumerate(sorted(candidates, key=lambda item: item.score, reverse=True), start=1)
    }
    frame_index = build_frame_index(candidates)
    output = []
    for pred_index, row in enumerate(pred_rows):
        row_key = (video_key, pred_index)
        if row_key not in first_stage_rows:
            continue
        frame = as_int(row["frame"], "frame")
        local = local_candidates(candidates, frame_index, frame, args.proposal_window)
        local = [
            candidate
            for candidate in local
            if global_ranks[candidate_key(candidate)] <= args.rank_cap and not same_prediction(row, candidate)
        ]
        by_distance = sorted(local, key=lambda candidate: (abs(candidate.frame - frame), -candidate.score))
        by_score = sorted(local, key=lambda candidate: candidate.score, reverse=True)
        distance_ranks = {candidate_key(candidate): rank for rank, candidate in enumerate(by_distance, start=1)}
        score_ranks = {candidate_key(candidate): rank for rank, candidate in enumerate(by_score, start=1)}
        seen = set()
        merged = [*by_distance[: args.max_candidates_per_row], *by_score[: args.max_candidates_per_row]]
        row_feature_values = tuple(row_features(row, video, candidates, [15, 30, 60, 90]))
        for candidate in merged:
            key = candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidate_row = dict(row)
            candidate_row["frame"] = str(candidate.frame)
            candidate_row["fighter"] = candidate.fighter
            candidate_row["hand"] = candidate.hand
            candidate_row["target"] = candidate.target
            candidate_row_feature_values = tuple(row_features(candidate_row, video, candidates, [15, 30, 60, 90]))
            output.append(
                Proposal(
                    video_key=video_key,
                    pred_index=pred_index,
                    old_row=row,
                    candidate=candidate,
                    cluster_p=cluster_probs[row_key],
                    cluster_rank=cluster_rank[row_key],
                    global_rank=global_ranks[key],
                    local_distance_rank=distance_ranks[key],
                    local_score_rank=score_ranks[key],
                    local_size=len(local),
                    row_feature_values=row_feature_values,
                    candidate_row_feature_values=candidate_row_feature_values,
                )
            )
    return output


def build_frame_index(candidates: list[PunchCandidate]) -> dict[str, object]:
    frames = np.asarray([candidate.frame for candidate in candidates], dtype=np.int32)
    order = np.argsort(frames)
    return {"frames": frames[order], "order": order}


def local_candidates(
    candidates: list[PunchCandidate],
    frame_index: dict[str, object],
    frame: int,
    window: int,
) -> list[PunchCandidate]:
    frames: np.ndarray = frame_index["frames"]  # type: ignore[assignment]
    order: np.ndarray = frame_index["order"]  # type: ignore[assignment]
    lo = int(np.searchsorted(frames, frame - window, side="left"))
    hi = int(np.searchsorted(frames, frame + window, side="right"))
    return [candidates[int(index)] for index in order[lo:hi]]


def same_prediction(row: dict[str, str], candidate: PunchCandidate) -> bool:
    return (
        as_int(row["frame"], "frame") == candidate.frame
        and row["fighter"] == candidate.fighter
        and row["hand"] == candidate.hand
        and row["target"] == candidate.target
    )


def proposal_key(proposal: Proposal) -> tuple[str, int, int, str, str, str]:
    candidate = proposal.candidate
    return (
        proposal.video_key,
        proposal.pred_index,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
    )


def candidate_key(candidate: PunchCandidate) -> tuple[int, str, str, str]:
    return (candidate.frame, candidate.fighter, candidate.hand, candidate.target)


def proposal_features(proposal: Proposal, video: dict[str, str]) -> list[float]:
    row = proposal.old_row
    candidate = proposal.candidate
    old_frame = as_int(row["frame"], "frame")
    frame_count = as_int(video["frame_count"], "frame_count")
    signed_delta = candidate.frame - old_frame
    values = []
    for name in BASE_FEATURES:
        value = float(candidate.features.get(name, 0.0))
        if name == "forward_norm":
            value /= 1000.0
        values.append(value)
    values.extend(
        [
            candidate.score,
            float(np.log1p(max(0.0, candidate.score))),
            proposal.cluster_p,
            proposal.cluster_rank / 1188.0,
            proposal.global_rank / 10000.0,
            proposal.local_distance_rank / max(1.0, float(proposal.local_size)),
            proposal.local_score_rank / max(1.0, float(proposal.local_size)),
            proposal.local_size / 500.0,
            signed_delta / 90.0,
            abs(signed_delta) / 90.0,
            old_frame / max(1.0, float(frame_count)),
            candidate.frame / max(1.0, float(frame_count)),
            float(row["fighter"] == candidate.fighter),
            float(row["hand"] == candidate.hand),
            float(row["target"] == candidate.target),
            float(row["fighter"] != candidate.fighter),
            float(row["hand"] != candidate.hand),
            float(row["target"] != candidate.target),
            float(candidate.fighter == "red"),
            float(candidate.hand == "left"),
            float(candidate.target == "head"),
            float(row["fighter"] == "red"),
            float(row["hand"] == "left"),
            float(row["target"] == "head"),
            float(video["dataset_type"] == "training"),
            float(video["data_root"] == "бокс"),
            float(video["data_root"] == "Турнир Бокс"),
            float(video["data_root"] == "Турнир Бокс 2"),
        ]
    )
    values.extend(proposal.row_feature_values)
    values.extend(proposal.candidate_row_feature_values)
    values.extend(
        [
            candidate_value - row_value
            for row_value, candidate_value in zip(proposal.row_feature_values, proposal.candidate_row_feature_values)
        ]
    )
    return values


def single_replacement_deltas(
    keys: list[str],
    pred_by_key: dict[str, list[dict[str, str]]],
    gt_by_key: dict[str, list[dict[str, str]]],
    proposals_by_key: dict[str, list[Proposal]],
) -> np.ndarray:
    deltas = []
    baseline_by_key = {
        key: float(score_video(gt_by_key.get(key, []), pred_by_key[key])["final_score"])
        for key in keys
    }
    for key in keys:
        for proposal in proposals_by_key[key]:
            trial = [dict(row) for row in pred_by_key[key]]
            row = trial[proposal.pred_index]
            row["frame"] = str(proposal.candidate.frame)
            row["fighter"] = proposal.candidate.fighter
            row["hand"] = proposal.candidate.hand
            row["target"] = proposal.candidate.target
            score = float(score_video(gt_by_key.get(key, []), trial)["final_score"])
            deltas.append(score - baseline_by_key[key])
    return np.asarray(deltas, dtype=np.float32)


def fit_oof_scores(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    p = np.zeros(len(y), dtype=np.float32)
    if len(y) == 0:
        return p
    for group in sorted(set(groups)):
        valid = groups == group
        train = ~valid
        y_train = y[train]
        model = build_model(model_name)
        if model_name.endswith("reg"):
            weights = regression_weights(target[train])
            try:
                model.fit(x[train], target[train], sample_weight=weights)
            except TypeError:
                model.fit(x[train], target[train])
            p[valid] = model.predict(x[valid]).astype(np.float32)
        else:
            if len(np.unique(y_train)) < 2:
                p[valid] = float(y_train.mean()) if len(y_train) else 0.0
                continue
            weights = balanced_weights(y_train) if model_name == "hgb" else None
            if weights is None:
                model.fit(x[train], y_train)
            else:
                model.fit(x[train], y_train, sample_weight=weights)
            p[valid] = model.predict_proba(x[valid])[:, 1].astype(np.float32)
        print(
            f"fold={group} train={int(train.sum())} valid={int(valid.sum())} "
            f"train_pos={int(y_train.sum())}",
            flush=True,
        )
    return p


def build_model(model_name: str):
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=42,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=6,
            random_state=42,
        )
    if model_name == "et":
        return ExtraTreesClassifier(
            n_estimators=900,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=6,
            random_state=42,
        )
    if model_name == "hgbreg":
        return HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=42,
        )
    if model_name == "rfreg":
        return RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=6,
            random_state=42,
        )
    if model_name == "etreg":
        return ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=6,
            random_state=42,
        )
    raise ValueError(f"unknown model: {model_name}")


def balanced_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    positives = y == 1
    if positives.any():
        weights[positives] = max(1.0, float((~positives).sum()) / max(1.0, float(positives.sum())))
    return weights


def regression_weights(target: np.ndarray) -> np.ndarray:
    weights = np.ones(len(target), dtype=np.float32)
    weights[target > 1e-12] = 3.0
    return weights


def combine_scores(mode: str, proposals: list[Proposal], candidate_p: np.ndarray) -> np.ndarray:
    cluster = np.asarray([proposal.cluster_p for proposal in proposals], dtype=np.float32)
    if mode == "candidate":
        return candidate_p.copy()
    if mode == "product":
        return candidate_p * cluster
    if mode == "sqrt_product":
        return candidate_p * np.sqrt(np.maximum(cluster, 0.0))
    raise ValueError(f"unknown combine mode: {mode}")


def apply_replacements(
    args: argparse.Namespace,
    threshold: float,
    keys: list[str],
    pred_by_key: dict[str, list[dict[str, str]]],
    proposals_by_key: dict[str, list[Proposal]],
    candidate_p: np.ndarray,
    combined: np.ndarray,
    accepted: set[tuple[str, int, int, str, str, str]],
) -> tuple[list[dict[str, str]], list[tuple[Proposal, float, float]]]:
    score_by_key = {}
    prob_by_key = {}
    index = 0
    for key in keys:
        for proposal in proposals_by_key[key]:
            pkey = proposal_key(proposal)
            prob_by_key[pkey] = float(candidate_p[index])
            score_by_key[pkey] = float(combined[index])
            index += 1

    selected: list[tuple[Proposal, float, float]] = []
    selected_pred = set()
    selected_candidate = set()
    video_counts: dict[str, int] = {}
    ranked = sorted(
        (proposal for key in keys for proposal in proposals_by_key[key]),
        key=lambda proposal: (
            score_by_key.get(proposal_key(proposal), 0.0),
            prob_by_key.get(proposal_key(proposal), 0.0),
            proposal.cluster_p,
        ),
        reverse=True,
    )
    for proposal in ranked:
        pkey = proposal_key(proposal)
        score = score_by_key.get(pkey, 0.0)
        if score < threshold or len(selected) >= args.max_replacements_total:
            break
        old_key = (proposal.video_key, proposal.pred_index)
        candidate = candidate_key(proposal.candidate)
        if old_key in selected_pred or (proposal.video_key, candidate) in selected_candidate:
            continue
        if video_counts.get(proposal.video_key, 0) >= args.max_replacements_per_video:
            continue
        selected_pred.add(old_key)
        selected_candidate.add((proposal.video_key, candidate))
        video_counts[proposal.video_key] = video_counts.get(proposal.video_key, 0) + 1
        selected.append((proposal, prob_by_key.get(pkey, 0.0), score))

    rows = []
    selected_by_pred = {(proposal.video_key, proposal.pred_index): proposal for proposal, *_ in selected}
    for key in keys:
        for pred_index, row in enumerate(pred_by_key[key]):
            item = dict(row)
            proposal = selected_by_pred.get((key, pred_index))
            if proposal is not None:
                item["frame"] = str(proposal.candidate.frame)
                item["fighter"] = proposal.candidate.fighter
                item["hand"] = proposal.candidate.hand
                item["target"] = proposal.candidate.target
            rows.append(item)
    return rows, selected


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "fp": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def print_score(
    label: str,
    score: dict[str, object],
    wins: int,
    n_rows: int,
    n_replaced: int,
    oracle_hits: int,
    oracle_misses: int,
) -> None:
    summary = score_summary(score)
    print(
        f"{label}: macro={float(score['macro_score']):.6f},time={summary['time']:.6f},"
        f"fighter={summary['fighter']:.6f},fp={summary['fp']:.6f},wins={wins},n={n_rows},"
        f"replaced={n_replaced},oracle_hits={oracle_hits},oracle_misses={oracle_misses}"
    )


def print_candidate_metrics(
    y: np.ndarray,
    candidate_p: np.ndarray,
    combined: np.ndarray,
    target: np.ndarray,
) -> None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        print("candidate_auc=nan candidate_ap=nan combined_auc=nan combined_ap=nan")
        return
    print(
        f"candidate_auc={roc_auc_score(y, candidate_p):.6f} "
        f"candidate_ap={average_precision_score(y, candidate_p):.6f} "
        f"combined_auc={roc_auc_score(y, combined):.6f} "
        f"combined_ap={average_precision_score(y, combined):.6f} "
        f"positive_target_sum={float(target[target > 0].sum()):.6f}"
    )
    order = np.argsort(-combined)
    total_pos = max(1, int(y.sum()))
    for top_n in [10, 20, 40, 80, 120]:
        selected = order[: min(top_n, len(order))]
        tp = int(y[selected].sum())
        print(f"top_combined_{top_n}: n={len(selected)} tp={tp} precision={tp / max(1, len(selected)):.6f} recall={tp / total_pos:.6f}")


def print_sweep_row(row: dict[str, str]) -> None:
    print(
        f"threshold={row['threshold']}: macro={row['macro']} delta_macro={row['delta_macro']},"
        f"time={row['time']} delta_time={row['delta_time']},fighter={row['fighter']} "
        f"delta_fighter={row['delta_fighter']},fp={row['fp']} delta_fp={row['delta_fp']},"
        f"wins={row['wins']},replaced={row['replaced']},oracle_hits={row['oracle_hits']},"
        f"oracle_misses={row['oracle_misses']},pass_all={row['pass_all']}"
    )


def selected_rows(
    threshold: float,
    selected: list[tuple[Proposal, float, float]],
    accepted: set[tuple[str, int, int, str, str, str]],
) -> list[dict[str, object]]:
    output = []
    ranked = sorted(selected, key=lambda item: item[2], reverse=True)
    for rank, (proposal, candidate_p, score) in enumerate(ranked, start=1):
        row = proposal.old_row
        candidate = proposal.candidate
        output.append(
            {
                "threshold": f"{threshold:.8g}",
                "rank": rank,
                "video_key": proposal.video_key,
                "pred_index": proposal.pred_index,
                "old_frame": row["frame"],
                "new_frame": candidate.frame,
                "old_fighter": row["fighter"],
                "new_fighter": candidate.fighter,
                "old_hand": row["hand"],
                "new_hand": candidate.hand,
                "old_target": row["target"],
                "new_target": candidate.target,
                "cluster_p": f"{proposal.cluster_p:.8f}",
                "candidate_p": f"{candidate_p:.8f}",
                "score": f"{score:.8f}",
                "is_oracle_replacement": int(proposal_key(proposal) in accepted),
            }
        )
    return output


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
