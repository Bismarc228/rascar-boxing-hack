#!/usr/bin/env python3
"""Evaluate a supervised fix-vs-break gate for pose-rival fighter proposals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, PunchCandidate, apply_temporal_context, score_pose_tracks
from tools.evaluate_exchange_state_gate import fight_group, parse_floats, parse_ints, safe_video_wins
from tools.evaluate_fighter_rival_flip import best_candidate, index_candidates, rival_predicate_factory


DEFAULT_BASE = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_"
    "posehgb094_vitlogreg012_rows_20260522.csv"
)
COMPONENTS = {
    "time": "score_time",
    "fighter": "score_fighter",
    "fp_penalty": "fp_penalty",
}
POSE_FEATURE_NAMES = [
    "closing",
    "arm_forward",
    "reach",
    "proximity",
    "conf",
    "role_conf",
    "attacker_red_score",
    "attacker_blue_score",
    "attacker_color_margin",
    "target_dist_norm",
    "head_dist_norm",
    "body_dist_norm",
    "forward_norm",
]


@dataclass(frozen=True)
class MatchLabel:
    matched: bool
    scorable: bool
    gt_fighter: str
    base_wrong: bool


@dataclass(frozen=True)
class Proposal:
    row_index: int
    proposed_fighter: str
    self_score: float
    rival_score: float
    rival_frame: int
    rival_hand: str
    rival_target: str
    self_features: dict[str, float]
    rival_features: dict[str, float]
    outcome: str

    @property
    def diff(self) -> float:
        return self.rival_score - self.self_score

    @property
    def ratio(self) -> float:
        return self.rival_score / max(0.05, self.self_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--match-mode", default="same_hand")
    parser.add_argument("--ratio", type=float, default=1.6)
    parser.add_argument("--min-rival-score", type=float, default=0.0)
    parser.add_argument("--appearance-summary", type=Path)
    parser.add_argument("--dual-candidate", type=Path)
    parser.add_argument("--models", default="hgb,logreg")
    parser.add_argument("--top-ns", default="1,2,3,4,5,8,10,12,15,20,25,30,40,50,60,80,100")
    parser.add_argument("--prob-thresholds", default="0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5,0.6")
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-proposals", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_rows = [row for row in read_csv_rows(args.base) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in base_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, base_rows)
    baseline_components = component_means(baseline)
    labels = build_match_labels(base_rows, gt_rows)

    candidates_by_key = load_candidates(args, keys)
    proposals = build_proposals(args, base_rows, video_by_key, labels, candidates_by_key)
    if not proposals:
        raise RuntimeError("no rival proposals produced")
    proposal_rows = writeable_proposals(base_rows, proposals)
    write_csv_rows(args.output_proposals, proposal_rows, proposal_columns())
    appearance_by_key = load_appearance(args.appearance_summary)
    dual_flip_by_index = load_dual_flips(args.dual_candidate, base_rows)

    x, numeric_count = build_feature_matrix(
        base_rows,
        video_by_key,
        proposals,
        args.window,
        appearance_by_key,
        dual_flip_by_index,
    )
    groups = np.asarray([fight_group(video_by_key[base_rows[item.row_index]["video_key"]]) for item in proposals])
    y = np.asarray([item.outcome == "fix" for item in proposals], dtype=np.int8)

    summary_rows: list[dict[str, object]] = []
    for model_name in split_values(args.models):
        probabilities = oof_probabilities(model_name, x, numeric_count, y, groups)
        summary_rows.extend(
            evaluate_thresholds(
                model_name,
                args,
                base_rows,
                gt_rows,
                baseline,
                baseline_components,
                proposals,
                probabilities,
            )
        )
        for row, prob in zip(proposal_rows, probabilities):
            row[f"p_fix_{model_name}"] = f"{float(prob):.9f}"

    write_csv_rows(args.output_proposals, proposal_rows, proposal_columns() + [f"p_fix_{m}" for m in split_values(args.models)])
    summary_rows.sort(
        key=lambda row: (float(row["fighter_delta"]), float(row["macro_delta"]), -int(row["selected"])),
        reverse=True,
    )
    write_csv_rows(args.output_summary, summary_rows, summary_columns())

    print_baseline(baseline, baseline_components, len(base_rows), proposals)
    print(f"wrote_summary={args.output_summary}")
    print(f"wrote_proposals={args.output_proposals}")
    print_top(summary_rows, args.top_k)
    return 0


def load_candidates(args: argparse.Namespace, keys: list[str]) -> dict[str, dict[int, list[PunchCandidate]]]:
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    return {
        key: index_candidates(
            apply_temporal_context(score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config), config)
        )
        for key in keys
    }


def load_appearance(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row["video_key"]: row for row in read_csv_rows(path)}


def load_dual_flips(path: Path | None, base_rows: list[dict[str, str]]) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    dual_rows = [row for row in read_csv_rows(path) if row.get("clear") == "true"]
    if len(dual_rows) != len(base_rows):
        raise RuntimeError(f"base/dual row count mismatch: {len(base_rows)} != {len(dual_rows)}")
    output = {}
    for index, (base, dual) in enumerate(zip(base_rows, dual_rows)):
        for key in ["id", "video_key", "frame"]:
            if base.get(key, "") != dual.get(key, ""):
                raise RuntimeError(f"base/dual mismatch at row {index}: {base} vs {dual}")
        output[index] = int(dual["fighter"] != base["fighter"])
    return output


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


def build_proposals(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    labels: list[MatchLabel],
    candidates_by_key: dict[str, dict[int, list[PunchCandidate]]],
) -> list[Proposal]:
    del video_by_key
    predicate_factory = rival_predicate_factory(args.match_mode)
    proposals = []
    for index, row in enumerate(rows):
        frame = as_int(row["frame"], "frame")
        candidates = candidates_by_key[row["video_key"]]
        self_candidate = best_candidate(
            candidates,
            frame,
            args.window,
            lambda candidate, row=row: candidate.fighter == row["fighter"]
            and candidate.hand == row["hand"]
            and candidate.target == row["target"],
        )
        if self_candidate is None:
            self_candidate = best_candidate(
                candidates,
                frame,
                args.window,
                lambda candidate, row=row: candidate.fighter == row["fighter"],
            )
        rival = best_candidate(candidates, frame, args.window, predicate_factory(row))
        self_score = float(self_candidate.score) if self_candidate else 0.0
        if rival is None or rival.score < args.min_rival_score or rival.score < self_score * args.ratio:
            continue
        proposals.append(
            Proposal(
                row_index=index,
                proposed_fighter=rival.fighter,
                self_score=self_score,
                rival_score=float(rival.score),
                rival_frame=int(rival.frame),
                rival_hand=rival.hand,
                rival_target=rival.target,
                self_features=dict(self_candidate.features) if self_candidate is not None else {},
                rival_features=dict(rival.features),
                outcome=proposal_outcome(row, labels[index], rival.fighter),
            )
        )
    return proposals


def proposal_outcome(row: dict[str, str], label: MatchLabel, proposed_fighter: str) -> str:
    if not label.matched:
        return "unmatched"
    if not label.scorable:
        return "time_only"
    if label.base_wrong and proposed_fighter == label.gt_fighter:
        return "fix"
    if not label.base_wrong and proposed_fighter != label.gt_fighter:
        return "break"
    if label.base_wrong:
        return "still_wrong"
    return "neutral"


def build_feature_matrix(
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    proposals: list[Proposal],
    window: int,
    appearance_by_key: dict[str, dict[str, str]],
    dual_flip_by_index: dict[int, int],
) -> tuple[np.ndarray, int]:
    numeric = []
    categorical = []
    by_video = defaultdict(list)
    for proposal in proposals:
        by_video[rows[proposal.row_index]["video_key"]].append(proposal)
    local_counts = local_proposal_counts(rows, proposals)
    for proposal in proposals:
        row = rows[proposal.row_index]
        video = video_by_key[row["video_key"]]
        appearance = appearance_by_key.get(row["video_key"], {})
        frame = as_int(row["frame"], "frame")
        frame_count = max(1, as_int(video["frame_count"], "frame_count"))
        numeric.append(
            [
                float(proposal.self_score),
                float(proposal.rival_score),
                float(proposal.diff),
                float(proposal.ratio),
                float(abs(proposal.rival_frame - frame)),
                float(frame / frame_count),
                float(frame_count),
                float(safe_int(video.get("round_number", ""))),
                float(window),
                float(dual_flip_by_index.get(proposal.row_index, 0)),
                parse_optional_float(appearance.get("color_sep_ratio", "")),
                parse_optional_float(appearance.get("color_nn_acc", "")),
                parse_optional_float(appearance.get("color_early_acc", "")),
                parse_optional_float(appearance.get("embedding_cos_dist", "")),
                parse_optional_float(appearance.get("embedding_sep_ratio", "")),
                parse_optional_float(appearance.get("embedding_nn_acc", "")),
                parse_optional_float(appearance.get("embedding_early_acc", "")),
                parse_optional_float(appearance.get("calibration_start_frame", "")),
                parse_optional_float(appearance.get("early_red_samples", "")),
                parse_optional_float(appearance.get("early_blue_samples", "")),
                *pose_feature_values(proposal),
                *local_counts[proposal.row_index],
            ]
        )
        categorical.append(
            [
                row["fighter"],
                row["hand"],
                row["target"],
                row["punch_type"],
                row["effectiveness"],
                proposal.rival_hand,
                proposal.rival_target,
                video.get("dataset_type", ""),
                video.get("data_root", ""),
                appearance.get("calibration_ok", ""),
                appearance.get("reason", ""),
            ]
        )
    numeric_arr = np.asarray(numeric, dtype=np.float32)
    categorical_arr = np.asarray(categorical, dtype=object)
    return np.concatenate([numeric_arr, categorical_arr], axis=1), numeric_arr.shape[1]


def pose_feature_values(proposal: Proposal) -> list[float]:
    values: list[float] = []
    for name in POSE_FEATURE_NAMES:
        self_value = parse_optional_float(str(proposal.self_features.get(name, "")))
        rival_value = parse_optional_float(str(proposal.rival_features.get(name, "")))
        values.extend(
            [
                self_value,
                rival_value,
                rival_value - self_value,
                safe_ratio(rival_value, self_value),
            ]
        )
    return values


def local_proposal_counts(rows: list[dict[str, str]], proposals: list[Proposal]) -> dict[int, list[float]]:
    by_video: dict[str, list[Proposal]] = defaultdict(list)
    for proposal in proposals:
        by_video[rows[proposal.row_index]["video_key"]].append(proposal)
    output = {}
    for items in by_video.values():
        for proposal in items:
            row = rows[proposal.row_index]
            frame = as_int(row["frame"], "frame")
            counts = Counter()
            for other in items:
                if other.row_index == proposal.row_index:
                    continue
                other_row = rows[other.row_index]
                distance = abs(as_int(other_row["frame"], "frame") - frame)
                if distance > 30:
                    continue
                counts["near30"] += 1
                counts["same_fighter"] += int(other_row["fighter"] == row["fighter"])
                counts["same_hand"] += int(other_row["hand"] == row["hand"])
                counts["same_type"] += int(other_row["punch_type"] == row["punch_type"])
                counts["same_eff"] += int(other_row["effectiveness"] == row["effectiveness"])
            output[proposal.row_index] = [
                float(counts["near30"]),
                float(counts["same_fighter"]),
                float(counts["same_hand"]),
                float(counts["same_type"]),
                float(counts["same_eff"]),
            ]
    return output


def oof_probabilities(model_name: str, x: np.ndarray, numeric_count: int, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    numeric_indices = list(range(numeric_count))
    categorical_indices = list(range(numeric_count, x.shape[1]))
    preprocessor = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_indices),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ]
    )
    if model_name == "hgb":
        estimator = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=23,
        )
    elif model_name == "logreg":
        estimator = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000, solver="lbfgs")
    else:
        raise ValueError(f"unknown model: {model_name}")
    model = make_pipeline(preprocessor, estimator)
    probabilities = np.zeros(len(y), dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = ~valid
        if len(np.unique(y[train])) < 2:
            probabilities[valid] = float(y[train].mean()) if train.any() else 0.0
            continue
        fold_model = clone(model)
        fold_model.fit(x[train], y[train])
        probabilities[valid] = fold_model.predict_proba(x[valid])[:, 1].astype(np.float32)
    return probabilities


def evaluate_thresholds(
    model_name: str,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    baseline: dict[str, object],
    baseline_components: dict[str, float],
    proposals: list[Proposal],
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    output = []
    order = np.argsort(-probabilities, kind="mergesort")
    for top_n in parse_ints(args.top_ns):
        if top_n < 1 or top_n > len(order):
            continue
        selected = {int(index) for index in order[:top_n]}
        output.append(
            score_selection(
                model_name,
                "top_n",
                f"{top_n}",
                rows,
                gt_rows,
                baseline,
                baseline_components,
                proposals,
                selected,
            )
        )
    for threshold in parse_floats(args.prob_thresholds):
        selected = {int(index) for index, value in enumerate(probabilities) if value >= threshold}
        if not selected:
            continue
        output.append(
            score_selection(
                model_name,
                "prob",
                f"{threshold:g}",
                rows,
                gt_rows,
                baseline,
                baseline_components,
                proposals,
                selected,
            )
        )
    return output


def score_selection(
    model_name: str,
    threshold_type: str,
    threshold: str,
    rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    baseline: dict[str, object],
    baseline_components: dict[str, float],
    proposals: list[Proposal],
    selected: set[int],
) -> dict[str, object]:
    selected_rows = apply_proposals(rows, proposals, selected)
    score = score_predictions(gt_rows, selected_rows)
    components = component_means(score)
    counts = Counter(proposals[index].outcome for index in selected)
    return {
        "model": model_name,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "selected": len(selected),
        "fix": counts["fix"],
        "break": counts["break"],
        "time_only": counts["time_only"],
        "unmatched": counts["unmatched"],
        "still_wrong": counts["still_wrong"],
        "macro_score": f"{float(score['macro_score']):.6f}",
        "macro_delta": f"{float(score['macro_score']) - float(baseline['macro_score']):+.6f}",
        "fighter": f"{components['fighter']:.6f}",
        "fighter_delta": f"{components['fighter'] - baseline_components['fighter']:+.6f}",
        "time": f"{components['time']:.6f}",
        "time_delta": f"{components['time'] - baseline_components['time']:+.6f}",
        "fp_penalty": f"{components['fp_penalty']:.6f}",
        "fp_penalty_delta": f"{components['fp_penalty'] - baseline_components['fp_penalty']:+.6f}",
        "wins": safe_video_wins(score, baseline),
        "min_video_delta": f"{min_video_delta(score, baseline):+.6f}",
    }


def apply_proposals(
    rows: list[dict[str, str]],
    proposals: list[Proposal],
    selected: set[int],
) -> list[dict[str, str]]:
    selected_by_row = {proposals[index].row_index: proposals[index].proposed_fighter for index in selected}
    output = []
    for index, row in enumerate(rows):
        item = {column: row.get(column, "") for column in SUBMISSION_COLUMNS}
        if index in selected_by_row:
            item["fighter"] = selected_by_row[index]
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


def writeable_proposals(rows: list[dict[str, str]], proposals: list[Proposal]) -> list[dict[str, object]]:
    output = []
    for index, proposal in enumerate(proposals):
        row = rows[proposal.row_index]
        output.append(
            {
                "proposal_index": index,
                "row_index": proposal.row_index,
                "id": row["id"],
                "video_key": row["video_key"],
                "frame": row["frame"],
                "fighter_before": row["fighter"],
                "fighter_after": proposal.proposed_fighter,
                "hand": row["hand"],
                "target": row["target"],
                "punch_type": row["punch_type"],
                "effectiveness": row["effectiveness"],
                "self_score": f"{proposal.self_score:.6f}",
                "rival_score": f"{proposal.rival_score:.6f}",
                "rival_diff": f"{proposal.diff:.6f}",
                "rival_ratio": f"{proposal.ratio:.6f}",
                "rival_frame": proposal.rival_frame,
                "rival_hand": proposal.rival_hand,
                "rival_target": proposal.rival_target,
                "outcome": proposal.outcome,
            }
        )
    return output


def proposal_columns() -> list[str]:
    return [
        "proposal_index",
        "row_index",
        "id",
        "video_key",
        "frame",
        "fighter_before",
        "fighter_after",
        "hand",
        "target",
        "punch_type",
        "effectiveness",
        "self_score",
        "rival_score",
        "rival_diff",
        "rival_ratio",
        "rival_frame",
        "rival_hand",
        "rival_target",
        "outcome",
    ]


def summary_columns() -> list[str]:
    return [
        "model",
        "threshold_type",
        "threshold",
        "selected",
        "fix",
        "break",
        "time_only",
        "unmatched",
        "still_wrong",
        "macro_score",
        "macro_delta",
        "fighter",
        "fighter_delta",
        "time",
        "time_delta",
        "fp_penalty",
        "fp_penalty_delta",
        "wins",
        "min_video_delta",
    ]


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_optional_float(value: str) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    return numerator / max(1e-6, abs(denominator))


def print_baseline(
    score: dict[str, object],
    components: dict[str, float],
    n_rows: int,
    proposals: list[Proposal],
) -> None:
    counts = Counter(item.outcome for item in proposals)
    print(
        f"baseline macro={float(score['macro_score']):.6f} "
        f"fighter={components['fighter']:.6f} time={components['time']:.6f} "
        f"fp_penalty={components['fp_penalty']:.6f} rows={n_rows} "
        f"proposals={len(proposals)} "
        + " ".join(f"{key}={counts[key]}" for key in sorted(counts)),
        flush=True,
    )


def print_top(rows: list[dict[str, object]], limit: int) -> None:
    print(
        "model,threshold_type,threshold,selected,fix,break,time_only,macro_delta,"
        "fighter_delta,time_delta,fp_penalty_delta,wins,min_video_delta"
    )
    for row in rows[:limit]:
        print(
            ",".join(
                str(row[column])
                for column in [
                    "model",
                    "threshold_type",
                    "threshold",
                    "selected",
                    "fix",
                    "break",
                    "time_only",
                    "macro_delta",
                    "fighter_delta",
                    "time_delta",
                    "fp_penalty_delta",
                    "wins",
                    "min_video_delta",
                ]
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
