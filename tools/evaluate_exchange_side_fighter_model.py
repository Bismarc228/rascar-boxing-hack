#!/usr/bin/env python3
"""Learn a high-precision fixed-row opposite-fighter flip model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    score_pose_tracks,
)
from tools.evaluate_exchange_state_gate import fight_group
from tools.evaluate_fighter_rival_flip import best_candidate, index_candidates, score_summary
from tools.evaluate_pose_selection_variants import video_wins


BASE_FEATURES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--windows", default="0,2")
    parser.add_argument("--match-modes", default="same_hand_target,same_hand")
    parser.add_argument("--update-modes", default="fighter_only")
    parser.add_argument("--models", default="hgb,logreg")
    parser.add_argument("--thresholds", default="0.70,0.80,0.90,0.95,0.98")
    parser.add_argument("--feature-windows", default="2,4,8")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-model", default="hgb")
    parser.add_argument("--write-window", type=int)
    parser.add_argument("--write-match-mode")
    parser.add_argument("--write-update-mode", default="fighter_only")
    parser.add_argument("--write-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
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
    indexed_by_key = {
        key: index_candidates(
            apply_temporal_context(score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config), config)
        )
        for key in keys
    }
    labels = flip_labels(pred_rows, gt_rows)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    windows = parse_ints(args.windows)
    feature_windows = parse_ints(args.feature_windows)

    results: list[dict[str, Any]] = []
    probabilities_by_config: dict[tuple[str, int, str], np.ndarray] = {}
    rivals_by_config: dict[tuple[int, str], list[PunchCandidate | None]] = {}
    for window in windows:
        for match_mode in parse_list(args.match_modes):
            x, action_mask, rivals = build_features(
                pred_rows,
                video_by_key,
                indexed_by_key,
                window,
                match_mode,
                feature_windows,
            )
            rivals_by_config[(window, match_mode)] = rivals
            y = labels.astype(np.int8)
            for model_name in parse_list(args.models):
                probabilities = fit_oof_probabilities(model_name, x, y, action_mask, groups)
                probabilities_by_config[(model_name, window, match_mode)] = probabilities
                print_label_summary(model_name, window, match_mode, y, action_mask, probabilities)
                for update_mode in parse_list(args.update_modes):
                    for threshold in parse_floats(args.thresholds):
                        rows, changed = apply_flips(pred_rows, rivals, probabilities >= threshold, update_mode)
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
                                "model": model_name,
                                "window": window,
                                "match_mode": match_mode,
                                "update_mode": update_mode,
                                "threshold": threshold,
                            }
                        )

    print(
        "score,delta,fighter,time,hand,target,fp,wins,n_changed,"
        "model,window,match_mode,update_mode,threshold"
    )
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print(
            f"{item['score']:.6f},{item['delta']:.6f},{item['fighter']:.6f},"
            f"{item['time']:.6f},{item['hand']:.6f},{item['target']:.6f},"
            f"{item['fp']:.6f},{item['wins']},{item['changed']},"
            f"{item['model']},{item['window']},{item['match_mode']},"
            f"{item['update_mode']},{item['threshold']}"
        )

    if args.write_oof_rows:
        if args.write_window is None or not args.write_match_mode or args.write_threshold is None:
            best = max(results, key=lambda row: row["score"])
            model_name = str(best["model"])
            window = int(best["window"])
            match_mode = str(best["match_mode"])
            update_mode = str(best["update_mode"])
            threshold = float(best["threshold"])
        else:
            model_name = args.write_model
            window = args.write_window
            match_mode = args.write_match_mode
            update_mode = args.write_update_mode
            threshold = args.write_threshold
        probabilities = probabilities_by_config[(model_name, window, match_mode)]
        rivals = rivals_by_config[(window, match_mode)]
        rows, changed = apply_flips(pred_rows, rivals, probabilities >= threshold, update_mode)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} model={model_name} window={window} "
            f"match_mode={match_mode} update_mode={update_mode} threshold={threshold} changed={changed}",
            flush=True,
        )
    return 0


def flip_labels(pred_rows: list[dict[str, str]], gt_rows: list[dict[str, str]]) -> np.ndarray:
    labels = np.zeros(len(pred_rows), dtype=np.int8)
    gt_by_key = group_by(gt_rows, "video_key")
    pred_by_key = group_by(pred_rows, "video_key")
    offset = 0
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            pred = rows[match.pred_index]
            gt = gt_by_key[key][match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) / FPS / 0.5 >= 1.0:
                continue
            if pred["fighter"] != gt["fighter"]:
                labels[offset + match.pred_index] = 1
        offset += len(rows)
    return labels


def build_features(
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    indexed_by_key: dict[str, dict[int, list[PunchCandidate]]],
    window: int,
    match_mode: str,
    feature_windows: list[int],
) -> tuple[np.ndarray, np.ndarray, list[PunchCandidate | None]]:
    x = []
    action_mask = []
    rivals = []
    for row in rows:
        candidates = indexed_by_key[row["video_key"]]
        frame = as_int(row["frame"], "frame")
        self_candidate = best_self_candidate(candidates, frame, window, row)
        rival = best_candidate(candidates, frame, window, rival_predicate(row, match_mode))
        rivals.append(rival)
        action_mask.append(rival is not None)
        x.append(row_features(row, video_by_key[row["video_key"]], candidates, self_candidate, rival, feature_windows))
    return np.stack(x).astype(np.float32), np.asarray(action_mask, dtype=bool), rivals


def best_self_candidate(
    candidates: dict[int, list[PunchCandidate]],
    frame: int,
    window: int,
    row: dict[str, str],
) -> PunchCandidate | None:
    self_candidate = best_candidate(
        candidates,
        frame,
        window,
        lambda candidate: candidate.fighter == row["fighter"]
        and candidate.hand == row["hand"]
        and candidate.target == row["target"],
    )
    if self_candidate is not None:
        return self_candidate
    self_candidate = best_candidate(
        candidates,
        frame,
        window,
        lambda candidate: candidate.fighter == row["fighter"] and candidate.hand == row["hand"],
    )
    if self_candidate is not None:
        return self_candidate
    return best_candidate(candidates, frame, window, lambda candidate: candidate.fighter == row["fighter"])


def rival_predicate(row: dict[str, str], match_mode: str):
    if match_mode == "same_hand_target":
        return (
            lambda candidate: candidate.fighter != row["fighter"]
            and candidate.hand == row["hand"]
            and candidate.target == row["target"]
        )
    if match_mode == "same_hand":
        return lambda candidate: candidate.fighter != row["fighter"] and candidate.hand == row["hand"]
    if match_mode == "any":
        return lambda candidate: candidate.fighter != row["fighter"]
    raise ValueError(f"Unknown match mode: {match_mode}")


def row_features(
    row: dict[str, str],
    video: dict[str, str],
    candidates_by_frame: dict[int, list[PunchCandidate]],
    self_candidate: PunchCandidate | None,
    rival: PunchCandidate | None,
    feature_windows: list[int],
) -> np.ndarray:
    frame = as_int(row["frame"], "frame")
    self_score = float(self_candidate.score) if self_candidate else 0.0
    rival_score = float(rival.score) if rival else 0.0
    frame_delta = float((rival.frame - frame) if rival else 99)
    values = [
        self_score,
        rival_score,
        float(np.log1p(max(0.0, self_score))),
        float(np.log1p(max(0.0, rival_score))),
        rival_score / max(1e-6, self_score),
        self_score - rival_score,
        frame_delta,
        abs(frame_delta),
        float(frame / max(1.0, float(video["frame_count"]))),
        1.0 if row["fighter"] == "red" else 0.0,
        1.0 if row["hand"] == "left" else 0.0,
        1.0 if row["target"] == "head" else 0.0,
    ]
    for root in ["бокс", "Турнир Бокс", "Турнир Бокс 2"]:
        values.append(float(video["data_root"] == root))
    values.extend(candidate_base_features(self_candidate))
    values.extend(candidate_base_features(rival))
    values.extend(candidate_feature_diff(self_candidate, rival))
    for current_window in feature_windows:
        values.extend(window_stats(candidates_by_frame, frame, current_window, row))
    return np.asarray(values, dtype=np.float32)


def candidate_base_features(candidate: PunchCandidate | None) -> list[float]:
    output = []
    for name in BASE_FEATURES:
        value = float(candidate.features.get(name, 0.0)) if candidate else 0.0
        if name == "forward_norm":
            value /= 1000.0
        output.append(value)
    return output


def candidate_feature_diff(self_candidate: PunchCandidate | None, rival: PunchCandidate | None) -> list[float]:
    self_values = candidate_base_features(self_candidate)
    rival_values = candidate_base_features(rival)
    return [rival_value - self_value for self_value, rival_value in zip(self_values, rival_values)]


def window_stats(
    candidates_by_frame: dict[int, list[PunchCandidate]],
    frame: int,
    window: int,
    row: dict[str, str],
) -> list[float]:
    same_scores = []
    rival_scores = []
    same_hand_rival_scores = []
    same_hand_target_rival_scores = []
    all_scores = []
    for current in range(frame - window, frame + window + 1):
        for candidate in candidates_by_frame.get(current, []):
            all_scores.append(candidate.score)
            if candidate.fighter == row["fighter"]:
                same_scores.append(candidate.score)
            else:
                rival_scores.append(candidate.score)
                if candidate.hand == row["hand"]:
                    same_hand_rival_scores.append(candidate.score)
                    if candidate.target == row["target"]:
                        same_hand_target_rival_scores.append(candidate.score)
    return [
        float(sum(all_scores)),
        float(len(all_scores)),
        max(all_scores) if all_scores else 0.0,
        float(sum(same_scores)),
        float(len(same_scores)),
        max(same_scores) if same_scores else 0.0,
        float(sum(rival_scores)),
        float(len(rival_scores)),
        max(rival_scores) if rival_scores else 0.0,
        float(sum(same_hand_rival_scores)),
        float(len(same_hand_rival_scores)),
        max(same_hand_rival_scores) if same_hand_rival_scores else 0.0,
        float(sum(same_hand_target_rival_scores)),
        float(len(same_hand_target_rival_scores)),
        max(same_hand_target_rival_scores) if same_hand_target_rival_scores else 0.0,
    ]


def fit_oof_probabilities(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    action_mask: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    output = np.zeros(len(y), dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        train = (groups != group) & action_mask
        valid = groups == group
        if train.sum() < 20 or len(np.unique(y[train])) < 2:
            output[valid] = 0.0
            continue
        model = make_model(model_name)
        model.fit(x[train], y[train])
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x[valid])[:, 1]
        else:
            prob = model.decision_function(x[valid])
        output[valid] = prob.astype(np.float32)
    output[~action_mask] = 0.0
    return output


def make_model(model_name: str):
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=20260521,
        )
    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.15),
        )
    raise ValueError(f"Unknown model: {model_name}")


def apply_flips(
    rows: list[dict[str, str]],
    rivals: list[PunchCandidate | None],
    flip_mask: np.ndarray,
    update_mode: str,
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for row, rival, flip in zip(rows, rivals, flip_mask):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if flip and rival is not None:
            if update_mode == "fighter_only":
                item["fighter"] = rival.fighter
            elif update_mode == "fighter_hand_target":
                item["fighter"] = rival.fighter
                item["hand"] = rival.hand
                item["target"] = rival.target
            else:
                raise ValueError(f"Unknown update mode: {update_mode}")
        if any(item[col] != row.get(col, "") for col in ["fighter", "hand", "target"]):
            changed += 1
        output.append(item)
    return output, changed


def print_label_summary(
    model_name: str,
    window: int,
    match_mode: str,
    y: np.ndarray,
    action_mask: np.ndarray,
    p: np.ndarray,
) -> None:
    action_y = y[action_mask]
    action_p = p[action_mask]
    pos_p = action_p[action_y == 1]
    neg_p = action_p[action_y == 0]
    print(
        f"label_summary,model={model_name},window={window},match_mode={match_mode},"
        f"n={int(action_mask.sum())},pos={int(action_y.sum())},"
        f"pos_rate={float(action_y.mean()) if len(action_y) else 0.0:.4f},"
        f"mean_p={float(action_p.mean()) if len(action_p) else 0.0:.4f},"
        f"pos_mean_p={float(pos_p.mean()) if len(pos_p) else 0.0:.4f},"
        f"neg_mean_p={float(neg_p.mean()) if len(neg_p) else 0.0:.4f}",
        flush=True,
    )


def print_score(label: str, score: dict[str, object], wins: int, n_rows: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},fighter={summary['fighter']:.6f},"
        f"time={summary['time']:.6f},fp={summary['fp']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
