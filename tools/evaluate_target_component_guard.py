#!/usr/bin/env python3
"""Evaluate guarded target-only fixed-row pose corrections."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, PunchCandidate, apply_temporal_context, score_pose_tracks
from tools.evaluate_exchange_state_gate import index_candidates, parse_floats, parse_ints
from tools.evaluate_fighter_rival_flip import best_candidate
from tools.evaluate_fixed_row_attribute_model import (
    attribute_features,
    fight_group,
    matched_attribute_labels,
)
from tools.evaluate_pose_selection_variants import video_wins


TARGET_VALUES = ["body", "head"]
HAND_VALUES = ["left", "right"]
EFFECTIVENESS_MODES = ["all", "landed", "nonlanded", "blocked", "miss"]
TRANSITION_MODES = ["both", "body_to_head", "head_to_body"]
PUBLIC_POLICIES = ["all", "exclude_public"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        action="append",
        required=True,
        help="Pose-track directory. Repeat to require stability across pose sources.",
    )
    parser.add_argument(
        "--feature-tracks-dir",
        type=Path,
        help="Pose-track directory for learned features; defaults to the first --tracks-dir.",
    )
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--prob-thresholds", default="0,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    parser.add_argument("--margin-thresholds", default="0,0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--stable-votes", default="0,1,2,3")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--exact-top-k", type=int, default=200)
    parser.add_argument("--public-sensitive-videos", default="agn_037,agn_038,agn_039")
    parser.add_argument("--diagnostics-prefix", type=Path)
    parser.add_argument("--write-selected-rows", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    if not pred_rows:
        raise RuntimeError(f"No clear rows in {args.predictions}")
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    missing = sorted(set(keys) - set(video_by_key))
    if missing:
        raise RuntimeError(f"Prediction keys are not train videos: {missing}")
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, pred_rows)
    base_attr = attr_summary(baseline)
    print_score("baseline", baseline, pred_rows, baseline, hand_changed=0)

    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    feature_tracks_dir = args.feature_tracks_dir or args.tracks_dir[0]
    feature_indexes = load_candidate_indexes(keys, feature_tracks_dir, config)
    windows = parse_ints(args.feature_windows)
    features = np.stack(
        [
            attribute_features(
                row,
                video_by_key[row["video_key"]],
                feature_indexes[row["video_key"]],
                args.match_window,
                windows,
            )
            for row in pred_rows
        ]
    )
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    target_pred, target_prob = oof_column_predictions("target", TARGET_VALUES, features, labels, groups)
    hand_pred, _ = oof_column_predictions("hand", HAND_VALUES, features, labels, groups)

    pose_indexes = [load_candidate_indexes(keys, tracks_dir, config) for tracks_dir in args.tracks_dir]
    pose_stats = build_pose_stats(pred_rows, pose_indexes, args.match_window)
    target_gain = build_target_gain(pred_rows, gt_rows, target_pred)
    public_sensitive = split_values(args.public_sensitive_videos)

    candidates = rank_guard_candidates(
        args,
        pred_rows,
        target_pred,
        target_prob,
        pose_stats,
        target_gain,
        baseline,
        base_attr["target"],
        public_sensitive,
    )
    exact_results = score_top_candidates(
        candidates[: args.exact_top_k],
        pred_rows,
        target_pred,
        gt_rows,
        baseline,
        video_by_key,
        public_sensitive,
    )
    if not exact_results:
        raise RuntimeError("No active target guard candidates were produced")
    selected = max(exact_results, key=lambda item: (item["macro"], item["target"], -item["changed"]))
    selected_rows = apply_target_mask(pred_rows, target_pred, selected["mask"])
    hand_changed = count_column_changes(pred_rows, selected_rows, "hand")
    fixed_ok = assert_fixed_target_only(pred_rows, selected_rows)
    root_rows = root_deltas(video_by_key, keys, baseline, selected["score"])
    tournament_non_negative = all(
        row["delta_macro"] >= -1e-12 for row in root_rows if str(row["dataset_type"]).startswith("tournament")
    )
    target_pass = (
        (selected["target_delta"] >= 0.020 or selected["delta"] >= 0.0012)
        and hand_changed == 0
        and fixed_ok
        and tournament_non_negative
    )

    negative_rows = apply_hand_control(pred_rows, hand_pred)
    negative_score = score_predictions(gt_rows, negative_rows)
    negative_hand_changed = count_column_changes(pred_rows, negative_rows, "hand")
    print_score("negative_control_hand", negative_score, negative_rows, baseline, negative_hand_changed)

    print("top_exact")
    print(
        "rank,macro,delta,target,target_delta,hand,hand_delta,wins,changed,"
        "effectiveness_mode,transition_mode,min_prob,min_margin,min_stable_votes,"
        "require_pose_majority,require_pose_change,public_policy,public_changed"
    )
    for rank, item in enumerate(exact_results[: args.top_k], start=1):
        print(format_result(rank, item, baseline, base_attr))

    print("selected")
    print(format_result(1, selected, baseline, base_attr))
    print(
        f"hand_unchanged_assertion={hand_changed == 0} fixed_target_only_assertion={fixed_ok} "
        f"tournament_roots_non_negative={tournament_non_negative} pass={target_pass}"
    )

    if args.write_selected_rows:
        write_csv_rows(args.write_selected_rows, selected_rows, SUBMISSION_COLUMNS)
        print(f"wrote_selected_rows={args.write_selected_rows} n_rows={len(selected_rows)}")

    prefix = args.diagnostics_prefix
    if prefix:
        write_diagnostics(
            prefix,
            exact_results,
            selected,
            pred_rows,
            selected_rows,
            gt_rows,
            baseline,
            video_by_key,
            keys,
            root_rows,
            negative_score,
            negative_hand_changed,
            target_pass,
        )
        print(f"wrote_diagnostics_prefix={prefix}")
    return 0


def load_candidate_indexes(
    keys: list[str],
    tracks_dir: Path,
    config: PoseHeuristicConfig,
) -> dict[str, dict[int, list[PunchCandidate]]]:
    return {
        key: index_candidates(
            apply_temporal_context(score_pose_tracks(tracks_dir / f"{key}.jsonl", config), config)
        )
        for key in keys
    }


def oof_column_predictions(
    column: str,
    values: list[str],
    features: np.ndarray,
    labels: list[dict[str, str] | None],
    groups: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    predictions = np.asarray([""] * len(labels), dtype=object)
    probabilities = [{value: 0.0 for value in values} for _ in labels]
    trainable = np.asarray([label is not None for label in labels], dtype=bool)
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = (groups != group) & trainable
        if train.sum() < 10:
            continue
        encoder = LabelEncoder()
        y = encoder.fit_transform([labels[index][column] for index in np.where(train)[0]])  # type: ignore[index]
        if len(encoder.classes_) == 1:
            predictions[valid] = encoder.classes_[0]
            for index in np.where(valid)[0]:
                probabilities[index][str(encoder.classes_[0])] = 1.0
            continue
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.3,
            random_state=42,
        )
        model.fit(features[train], y)
        encoded = model.predict(features[valid])
        proba = model.predict_proba(features[valid])
        predictions[valid] = encoder.inverse_transform(encoded)
        for index, row_proba in zip(np.where(valid)[0], proba):
            probabilities[index].update(
                {str(name): float(value) for name, value in zip(encoder.classes_, row_proba)}
            )
        print(
            f"fold={group} column={column} train={int(train.sum())} valid={int(valid.sum())}",
            flush=True,
        )
    for index, value in enumerate(predictions):
        if value == "":
            fallback = labels[index][column] if labels[index] else ""
            predictions[index] = fallback
            if fallback:
                probabilities[index][str(fallback)] = max(1.0, probabilities[index].get(str(fallback), 0.0))
    return predictions, probabilities


def build_pose_stats(
    rows: list[dict[str, str]],
    pose_indexes: list[dict[str, dict[int, list[PunchCandidate]]]],
    match_window: int,
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        frame = as_int(row["frame"], "frame")
        votes = []
        scores = []
        for indexes in pose_indexes:
            candidate = best_candidate(
                indexes[row["video_key"]],
                frame,
                match_window,
                lambda item, row=row: item.fighter == row["fighter"] and item.hand == row["hand"],
            )
            if candidate is None:
                continue
            votes.append(candidate.target)
            scores.append(float(candidate.score))
        counts = Counter(votes)
        majority, majority_count = ("", 0) if not counts else counts.most_common(1)[0]
        output.append(
            {
                "pose_votes": "|".join(votes),
                "pose_majority": majority,
                "pose_majority_count": majority_count,
                "pose_sources": len(votes),
                "pose_min_score": min(scores) if scores else 0.0,
                "pose_mean_score": float(np.mean(scores)) if scores else 0.0,
            }
        )
    return output


def build_target_gain(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    target_pred: np.ndarray,
) -> np.ndarray:
    gain = np.zeros(len(pred_rows), dtype=np.float64)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    global_indices_by_key: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        global_indices_by_key[row["video_key"]].append(index)
    for key, rows in pred_by_key.items():
        gt_video = gt_by_key.get(key, [])
        n_gt = max(1, len(gt_video))
        for match in match_events(gt_video, rows):
            pred = rows[match.pred_index]
            gt = gt_video[match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) / FPS / 0.5 >= 1.0:
                continue
            index = global_indices_by_key[key][match.pred_index]
            old_ok = pred["target"] == gt["target"]
            new_ok = str(target_pred[index]) == gt["target"]
            gain[index] = (int(new_ok) - int(old_ok)) / n_gt
    return gain


def rank_guard_candidates(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    target_pred: np.ndarray,
    target_prob: list[dict[str, float]],
    pose_stats: list[dict[str, object]],
    target_gain: np.ndarray,
    baseline: dict[str, Any],
    base_target: float,
    public_sensitive: set[str],
) -> list[dict[str, Any]]:
    keys = sorted({row["video_key"] for row in rows})
    prob_thresholds = parse_floats(args.prob_thresholds)
    margin_thresholds = parse_floats(args.margin_thresholds)
    stable_votes = parse_ints(args.stable_votes)
    candidates = []
    for effectiveness_mode in EFFECTIVENESS_MODES:
        effect_predicate = effectiveness_predicate(effectiveness_mode)
        for transition_mode in TRANSITION_MODES:
            transition_predicate = transition_predicate_factory(transition_mode)
            for min_prob in prob_thresholds:
                for min_margin in margin_thresholds:
                    for min_stable_votes in stable_votes:
                        for require_pose_majority in [False, True]:
                            for require_pose_change in [False, True]:
                                for public_policy in PUBLIC_POLICIES:
                                    mask = build_guard_mask(
                                        rows,
                                        target_pred,
                                        target_prob,
                                        pose_stats,
                                        effect_predicate,
                                        transition_predicate,
                                        min_prob,
                                        min_margin,
                                        min_stable_votes,
                                        require_pose_majority,
                                        require_pose_change,
                                        public_policy,
                                        public_sensitive,
                                    )
                                    changed = int(mask.sum())
                                    if changed == 0:
                                        continue
                                    target_delta = target_component_delta(rows, keys, mask, target_gain)
                                    candidates.append(
                                        {
                                            "approx_macro": float(baseline["macro_score"] + 0.06 * target_delta),
                                            "approx_target": float(base_target + target_delta),
                                            "target_delta": float(target_delta),
                                            "changed": changed,
                                            "effectiveness_mode": effectiveness_mode,
                                            "transition_mode": transition_mode,
                                            "min_prob": min_prob,
                                            "min_margin": min_margin,
                                            "min_stable_votes": min_stable_votes,
                                            "require_pose_majority": require_pose_majority,
                                            "require_pose_change": require_pose_change,
                                            "public_policy": public_policy,
                                            "public_changed": count_public_changes(rows, mask, public_sensitive),
                                            "mask": mask,
                                        }
                                    )
    return sorted(
        candidates,
        key=lambda item: (
            item["approx_macro"],
            item["approx_target"],
            -item["changed"],
            item["require_pose_majority"],
            item["min_stable_votes"],
        ),
        reverse=True,
    )


def build_guard_mask(
    rows: list[dict[str, str]],
    target_pred: np.ndarray,
    target_prob: list[dict[str, float]],
    pose_stats: list[dict[str, object]],
    effect_predicate: Callable[[dict[str, str]], bool],
    transition_predicate: Callable[[dict[str, str], str], bool],
    min_prob: float,
    min_margin: float,
    min_stable_votes: int,
    require_pose_majority: bool,
    require_pose_change: bool,
    public_policy: str,
    public_sensitive: set[str],
) -> np.ndarray:
    mask = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        predicted = str(target_pred[index])
        if not predicted or predicted == row["target"]:
            continue
        prob = target_prob[index].get(predicted, 0.0)
        original_prob = target_prob[index].get(row["target"], 0.0)
        stats = pose_stats[index]
        pose_majority = str(stats["pose_majority"])
        ok = (
            effect_predicate(row)
            and transition_predicate(row, predicted)
            and prob >= min_prob
            and prob - original_prob >= min_margin
            and int(stats["pose_majority_count"]) >= min_stable_votes
        )
        if require_pose_majority:
            ok = ok and pose_majority == predicted
        if require_pose_change:
            ok = ok and pose_majority != "" and pose_majority != row["target"]
        if public_policy == "exclude_public":
            if row["video_key"] in public_sensitive:
                ok = False
        elif public_policy != "all":
            raise ValueError(f"Unknown public policy: {public_policy}")
        mask[index] = ok
    return mask


def target_component_delta(
    rows: list[dict[str, str]],
    keys: list[str],
    mask: np.ndarray,
    target_gain: np.ndarray,
) -> float:
    total = 0.0
    for key in keys:
        indices = np.asarray([index for index, row in enumerate(rows) if row["video_key"] == key], dtype=int)
        if len(indices):
            total += float(target_gain[indices][mask[indices]].sum())
    return total / max(1, len(keys))


def score_top_candidates(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target_pred: np.ndarray,
    gt_rows: list[dict[str, str]],
    baseline: dict[str, Any],
    video_by_key: dict[str, dict[str, str]],
    public_sensitive: set[str],
) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for candidate in candidates:
        key = tuple(bool(value) for value in candidate["mask"].tolist())
        if key in seen:
            continue
        seen.add(key)
        modified = apply_target_mask(rows, target_pred, candidate["mask"])
        score = score_predictions(gt_rows, modified)
        attr = attr_summary(score)
        hand_changed = count_column_changes(rows, modified, "hand")
        item = dict(candidate)
        item.update(
            {
                "score": score,
                "macro": float(score["macro_score"]),
                "delta": float(score["macro_score"] - baseline["macro_score"]),
                "target": attr["target"],
                "target_delta": attr["target"] - attr_summary(baseline)["target"],
                "hand": attr["hand"],
                "hand_changed": hand_changed,
                "wins": video_wins(score, baseline),
                "tournament_non_negative": tournament_roots_non_negative(video_by_key, baseline, score),
                "public_changed": count_public_changes(rows, candidate["mask"], public_sensitive),
            }
        )
        output.append(item)
    return sorted(output, key=lambda item: (item["macro"], item["target"], -item["changed"]), reverse=True)


def apply_target_mask(
    rows: list[dict[str, str]],
    target_pred: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, str]]:
    output = []
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if mask[index] and str(target_pred[index]):
            item["target"] = str(target_pred[index])
        output.append(item)
    return output


def apply_hand_control(rows: list[dict[str, str]], hand_pred: np.ndarray) -> list[dict[str, str]]:
    output = []
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if str(hand_pred[index]):
            item["hand"] = str(hand_pred[index])
        output.append(item)
    return output


def assert_fixed_target_only(
    original_rows: list[dict[str, str]],
    modified_rows: list[dict[str, str]],
) -> bool:
    if len(original_rows) != len(modified_rows):
        return False
    protected = [column for column in SUBMISSION_COLUMNS if column != "target"]
    for old, new in zip(original_rows, modified_rows):
        for column in protected:
            if old.get(column, "") != new.get(column, ""):
                return False
    return True


def effectiveness_predicate(mode: str) -> Callable[[dict[str, str]], bool]:
    if mode == "all":
        return lambda row: True
    if mode == "landed":
        return lambda row: row["effectiveness"] == "landed"
    if mode == "nonlanded":
        return lambda row: row["effectiveness"] != "landed"
    if mode == "blocked":
        return lambda row: row["effectiveness"] == "blocked"
    if mode == "miss":
        return lambda row: row["effectiveness"] == "miss"
    raise ValueError(f"Unknown effectiveness mode: {mode}")


def transition_predicate_factory(mode: str) -> Callable[[dict[str, str], str], bool]:
    if mode == "both":
        return lambda row, predicted: True
    if mode == "body_to_head":
        return lambda row, predicted: row["target"] == "body" and predicted == "head"
    if mode == "head_to_body":
        return lambda row, predicted: row["target"] == "head" and predicted == "body"
    raise ValueError(f"Unknown transition mode: {mode}")


def root_deltas(
    video_by_key: dict[str, dict[str, str]],
    keys: list[str],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, object]]:
    roots: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        video = video_by_key[key]
        roots[(video["dataset_type"], video["data_root"])].append(key)
    output = []
    for (dataset_type, data_root), root_keys in sorted(roots.items()):
        base = aggregate_score(baseline, root_keys)
        cand = aggregate_score(candidate, root_keys)
        output.append(
            {
                "dataset_type": dataset_type,
                "data_root": data_root,
                "n_videos": len(root_keys),
                "baseline_macro": base["macro"],
                "candidate_macro": cand["macro"],
                "delta_macro": cand["macro"] - base["macro"],
                "baseline_target": base["target"],
                "candidate_target": cand["target"],
                "delta_target": cand["target"] - base["target"],
                "wins": sum(
                    candidate["by_video"][key]["final_score"] > baseline["by_video"][key]["final_score"] + 1e-12
                    for key in root_keys
                ),
                "losses": sum(
                    baseline["by_video"][key]["final_score"] > candidate["by_video"][key]["final_score"] + 1e-12
                    for key in root_keys
                ),
            }
        )
    return output


def video_deltas(
    video_by_key: dict[str, dict[str, str]],
    keys: list[str],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, object]]:
    output = []
    for key in keys:
        video = video_by_key[key]
        base = baseline["by_video"][key]
        cand = candidate["by_video"][key]
        output.append(
            {
                "video_key": key,
                "dataset_type": video["dataset_type"],
                "data_root": video["data_root"],
                "baseline_macro": base["final_score"],
                "candidate_macro": cand["final_score"],
                "delta_macro": cand["final_score"] - base["final_score"],
                "baseline_target": base["score_target"],
                "candidate_target": cand["score_target"],
                "delta_target": cand["score_target"] - base["score_target"],
                "n_gt": base["n_gt"],
                "n_pred": base["n_pred"],
                "result": "win"
                if cand["final_score"] > base["final_score"] + 1e-12
                else "loss"
                if base["final_score"] > cand["final_score"] + 1e-12
                else "tie",
            }
        )
    return output


def target_confusion_rows(
    label: str,
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    confusion: Counter[tuple[str, str]] = Counter()
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    for key, rows in pred_by_key.items():
        gt_video = gt_by_key.get(key, [])
        for match in match_events(gt_video, rows):
            pred = rows[match.pred_index]
            gt = gt_video[match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) / FPS / 0.5 >= 1.0:
                continue
            confusion[(gt["target"], pred["target"])] += 1
    return [
        {"variant": label, "actual_target": actual, "pred_target": predicted, "n": confusion[(actual, predicted)]}
        for actual in TARGET_VALUES
        for predicted in TARGET_VALUES
    ]


def change_rows(
    original_rows: list[dict[str, str]],
    modified_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    output = []
    for old, new in zip(original_rows, modified_rows):
        if old["target"] == new["target"]:
            continue
        output.append(
            {
                "id": old["id"],
                "video_key": old["video_key"],
                "frame": old["frame"],
                "fighter": old["fighter"],
                "hand": old["hand"],
                "effectiveness": old["effectiveness"],
                "old_target": old["target"],
                "new_target": new["target"],
            }
        )
    return output


def write_diagnostics(
    prefix: Path,
    exact_results: list[dict[str, Any]],
    selected: dict[str, Any],
    original_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    baseline: dict[str, Any],
    video_by_key: dict[str, dict[str, str]],
    keys: list[str],
    root_rows: list[dict[str, object]],
    negative_score: dict[str, Any],
    negative_hand_changed: int,
    target_pass: bool,
) -> None:
    summary_columns = [
        "rank",
        "macro",
        "delta",
        "target",
        "target_delta",
        "hand",
        "hand_delta",
        "wins",
        "changed",
        "effectiveness_mode",
        "transition_mode",
        "min_prob",
        "min_margin",
        "min_stable_votes",
        "require_pose_majority",
        "require_pose_change",
        "public_policy",
        "public_changed",
        "hand_changed",
        "tournament_non_negative",
    ]
    base_attr = attr_summary(baseline)
    rows = []
    for rank, item in enumerate(exact_results, start=1):
        rows.append(summary_item(rank, item, baseline, base_attr))
    write_csv_rows(prefix.with_name(prefix.name + "_summary.csv"), rows, summary_columns)

    write_csv_rows(
        prefix.with_name(prefix.name + "_changes.csv"),
        change_rows(original_rows, selected_rows),
        ["id", "video_key", "frame", "fighter", "hand", "effectiveness", "old_target", "new_target"],
    )
    confusion = target_confusion_rows("baseline", gt_rows, original_rows)
    confusion.extend(target_confusion_rows("selected", gt_rows, selected_rows))
    write_csv_rows(
        prefix.with_name(prefix.name + "_target_confusion.csv"),
        confusion,
        ["variant", "actual_target", "pred_target", "n"],
    )
    write_csv_rows(
        prefix.with_name(prefix.name + "_root_deltas.csv"),
        root_rows,
        [
            "dataset_type",
            "data_root",
            "n_videos",
            "baseline_macro",
            "candidate_macro",
            "delta_macro",
            "baseline_target",
            "candidate_target",
            "delta_target",
            "wins",
            "losses",
        ],
    )
    write_csv_rows(
        prefix.with_name(prefix.name + "_video_deltas.csv"),
        video_deltas(video_by_key, keys, baseline, selected["score"]),
        [
            "video_key",
            "dataset_type",
            "data_root",
            "baseline_macro",
            "candidate_macro",
            "delta_macro",
            "baseline_target",
            "candidate_target",
            "delta_target",
            "n_gt",
            "n_pred",
            "result",
        ],
    )
    negative_attr = attr_summary(negative_score)
    write_csv_rows(
        prefix.with_name(prefix.name + "_negative_control.csv"),
        [
            {
                "macro": negative_score["macro_score"],
                "delta": negative_score["macro_score"] - baseline["macro_score"],
                "hand": negative_attr["hand"],
                "hand_delta": negative_attr["hand"] - base_attr["hand"],
                "hand_changed": negative_hand_changed,
                "target_pass": target_pass,
            }
        ],
        ["macro", "delta", "hand", "hand_delta", "hand_changed", "target_pass"],
    )


def tournament_roots_non_negative(
    video_by_key: dict[str, dict[str, str]],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    keys = sorted(baseline["by_video"])
    for row in root_deltas(video_by_key, keys, baseline, candidate):
        if str(row["dataset_type"]).startswith("tournament") and row["delta_macro"] < -1e-12:
            return False
    return True


def aggregate_score(score: dict[str, Any], keys: list[str]) -> dict[str, float]:
    values = [score["by_video"][key] for key in keys]
    return {
        "macro": float(np.mean([item["final_score"] for item in values])),
        "target": float(np.mean([item["score_target"] for item in values])),
    }


def attr_summary(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "punch_type": float(np.mean([item["score_punch_type"] for item in by_video.values()])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
    }


def print_score(
    label: str,
    score: dict[str, Any],
    rows: list[dict[str, str]],
    baseline: dict[str, Any],
    hand_changed: int,
) -> None:
    summary = attr_summary(score)
    print(
        f"{label}: macro={score['macro_score']:.6f},delta={score['macro_score'] - baseline['macro_score']:.6f},"
        f"target={summary['target']:.6f},hand={summary['hand']:.6f},"
        f"wins={video_wins(score, baseline)},n={len(rows)},hand_changed={hand_changed}",
        flush=True,
    )


def format_result(
    rank: int,
    item: dict[str, Any],
    baseline: dict[str, Any],
    base_attr: dict[str, float],
) -> str:
    row = summary_item(rank, item, baseline, base_attr)
    return ",".join(str(row[column]) for column in [
        "rank",
        "macro",
        "delta",
        "target",
        "target_delta",
        "hand",
        "hand_delta",
        "wins",
        "changed",
        "effectiveness_mode",
        "transition_mode",
        "min_prob",
        "min_margin",
        "min_stable_votes",
        "require_pose_majority",
        "require_pose_change",
        "public_policy",
        "public_changed",
    ])


def summary_item(
    rank: int,
    item: dict[str, Any],
    baseline: dict[str, Any],
    base_attr: dict[str, float],
) -> dict[str, object]:
    return {
        "rank": rank,
        "macro": f"{item['macro']:.9f}",
        "delta": f"{item['delta']:.9f}",
        "target": f"{item['target']:.9f}",
        "target_delta": f"{item['target'] - base_attr['target']:.9f}",
        "hand": f"{item['hand']:.9f}",
        "hand_delta": f"{item['hand'] - base_attr['hand']:.9f}",
        "wins": item["wins"],
        "changed": item["changed"],
        "effectiveness_mode": item["effectiveness_mode"],
        "transition_mode": item["transition_mode"],
        "min_prob": item["min_prob"],
        "min_margin": item["min_margin"],
        "min_stable_votes": item["min_stable_votes"],
        "require_pose_majority": item["require_pose_majority"],
        "require_pose_change": item["require_pose_change"],
        "public_policy": item["public_policy"],
        "public_changed": item["public_changed"],
        "hand_changed": item["hand_changed"],
        "tournament_non_negative": item["tournament_non_negative"],
    }


def count_column_changes(
    original_rows: list[dict[str, str]],
    modified_rows: list[dict[str, str]],
    column: str,
) -> int:
    return sum(old[column] != new[column] for old, new in zip(original_rows, modified_rows))


def count_public_changes(
    rows: list[dict[str, str]],
    mask: np.ndarray,
    public_sensitive: set[str],
) -> int:
    return sum(int(keep and row["video_key"] in public_sensitive) for row, keep in zip(rows, mask))


def split_values(text: str) -> set[str]:
    return {value.strip() for value in text.split(",") if value.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
