#!/usr/bin/env python3
"""Evaluate learned fighter-label correction with fixed selected events."""

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    estimate_attrs,
    estimate_count,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from tools.evaluate_fighter_identity_smoothing import TrackInfo, load_track_info
from tools.evaluate_pose_selection_variants import score_summary, video_wins


@dataclass
class EventExample:
    key: str
    candidate: PunchCandidate
    x: np.ndarray
    y_correct: int | None


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
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--nms-frames", type=int, default=8)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--count-mode", default="threshold")
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    parser.add_argument("--context-feature", default="same_count")
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--flip-thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

    ready_set = set(ready_keys)
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    train_gt = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    attr_priors = fit_attr_priors(train_gt)
    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode="fighter_hand",
        cross_nms_frames=args.cross_nms_frames,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )

    gt_by_key = group_by_key(gt, "video_key")
    selected_by_key: dict[str, list[PunchCandidate]] = {}
    track_info_by_key: dict[str, TrackInfo] = {}
    examples_by_key: dict[str, list[EventExample]] = {}
    for key in ready_keys:
        tracks_path = args.tracks_dir / f"{key}.jsonl"
        candidates = score_pose_tracks(tracks_path, PoseHeuristicConfig(min_score=0.0))
        candidates = apply_temporal_context(candidates, config)
        count = estimate_count(config, video_by_key[key], train_videos, train_gt, [])
        selected = select_candidates(candidates, config, count)
        selected_by_key[key] = selected
        track_info_by_key[key] = load_track_info(tracks_path)
        base_rows = build_rows_for_key(video_by_key[key], selected, attr_priors, {})
        labels = matched_fighter_labels(gt_by_key.get(key, []), base_rows)
        examples_by_key[key] = [
            EventExample(
                key=key,
                candidate=candidate,
                x=build_features(video_by_key[key], candidate, track_info_by_key[key]),
                y_correct=labels.get(index),
            )
            for index, candidate in enumerate(selected)
        ]

    baseline_rows = build_rows(ready_keys, video_by_key, selected_by_key, attr_priors, {})
    baseline_score = score_predictions(gt, baseline_rows)
    oracle_rows = oracle_matched_rows(gt_by_key, ready_keys, video_by_key, selected_by_key, attr_priors)
    oracle_score = score_predictions(gt, oracle_rows)
    print_score("baseline", baseline_score, len(baseline_rows), 0)
    print_score("oracle_matched_fighter", oracle_score, len(oracle_rows), video_wins(oracle_score, baseline_score))

    group_by_video = {key: fight_group(video_by_key[key]) for key in ready_keys}
    groups = sorted(set(group_by_video.values()))
    p_correct_by_key: dict[str, np.ndarray] = {}
    for group in groups:
        valid_keys = [key for key in ready_keys if group_by_video[key] == group]
        train_keys = [key for key in ready_keys if group_by_video[key] != group]
        train_examples = [
            example
            for key in train_keys
            for example in examples_by_key[key]
            if example.y_correct is not None
        ]
        if not train_examples or len({example.y_correct for example in train_examples}) < 2:
            for key in valid_keys:
                p_correct_by_key[key] = np.ones(len(examples_by_key[key]), dtype=np.float32)
            continue
        x_train = np.stack([example.x for example in train_examples])
        y_train = np.array([example.y_correct for example in train_examples], dtype=np.int8)
        model = fit_model(args.model, x_train, y_train)
        for key in valid_keys:
            x_valid = np.stack([example.x for example in examples_by_key[key]])
            p_correct_by_key[key] = model.predict_proba(x_valid)[:, 1].astype(np.float32)
        print(
            f"fold={group} train_examples={len(train_examples)} "
            f"valid={','.join(valid_keys)} pos_rate={float(y_train.mean()):.4f}",
            flush=True,
        )

    results = []
    for threshold in parse_floats(args.flip_thresholds):
        flip_by_key = {
            key: {
                index
                for index, probability in enumerate(p_correct_by_key[key])
                if probability < threshold
            }
            for key in ready_keys
        }
        rows = build_rows(ready_keys, video_by_key, selected_by_key, attr_priors, flip_by_key)
        score = score_predictions(gt, rows)
        summary = score_summary_with_fighter(score)
        results.append(
            (
                score["macro_score"],
                summary["fighter"],
                summary["time"],
                summary["fp_penalty"],
                video_wins(score, baseline_score),
                sum(len(items) for items in flip_by_key.values()),
                threshold,
                score,
            )
        )

    print("score,fighter,time,fp,wins,n_flipped,threshold")
    for result in sorted(results, reverse=True)[: args.top_k]:
        print(
            f"{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},"
            f"{result[3]:.6f},{result[4]},{result[5]},{result[6]}"
        )
    best = max(results, key=lambda item: item[0])
    print_root_deltas(best[-1], baseline_score, ready_keys, video_by_key)
    return 0


def complete_track_keys(tracks_dir: Path, video_by_key: dict[str, dict[str, str]]) -> list[str]:
    keys = []
    for path in sorted(tracks_dir.glob("*.jsonl")):
        video = video_by_key.get(path.stem)
        if video is None:
            continue
        with path.open("r", encoding="utf-8") as fh:
            line_count = sum(1 for _ in fh)
        if line_count == int(video["frame_count"]):
            keys.append(path.stem)
    return keys


def build_features(video: dict[str, str], candidate: PunchCandidate, track_info: TrackInfo) -> np.ndarray:
    values = [
        float(np.log1p(max(0.0, candidate.score))),
        float(candidate.score),
        float(candidate.frame / max(1.0, float(video["frame_count"]))),
        1.0 if candidate.fighter == "red" else 0.0,
        1.0 if candidate.hand == "left" else 0.0,
        1.0 if candidate.target == "head" else 0.0,
    ]
    for root in ["бокс", "Турнир Бокс", "Турнир Бокс 2"]:
        values.append(float(video["data_root"] == root))
    for name in BASE_FEATURES:
        value = float(candidate.features.get(name, 0.0))
        if name == "forward_norm":
            value /= 1000.0
        values.append(value)
    track_id = track_info.frame_role_track.get((candidate.frame, candidate.fighter))
    role_majority = track_info.role_majority_by_track.get(track_id, candidate.fighter)
    color_mean = track_info.color_fighter_by_track.get(track_id, candidate.fighter)
    values.extend(
        [
            float(role_majority == candidate.fighter),
            float(color_mean == candidate.fighter),
            float(role_majority == color_mean),
            float(track_id if track_id is not None else -1) / 1000.0,
            abs(float(candidate.features.get("attacker_color_margin", 0.0))),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def matched_fighter_labels(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
) -> dict[int, int]:
    labels = {}
    for match in match_events(gt_rows, pred_rows):
        pred = pred_rows[match.pred_index]
        gt = gt_rows[match.gt_index]
        labels[match.pred_index] = int(pred["fighter"] == gt["fighter"])
    return labels


def fit_model(model_name: str, x: np.ndarray, y: np.ndarray):
    if model_name == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5),
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=42,
        )
    model.fit(x, y)
    return model


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, Any],
    flip_by_key: dict[str, set[int]],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        rows.extend(
            build_rows_for_key(
                video_by_key[key],
                selected_by_key[key],
                attr_priors,
                flip_by_key.get(key, set()),
                start_id=row_id,
            )
        )
        row_id += len(selected_by_key[key])
    return rows


def build_rows_for_key(
    video: dict[str, str],
    selected: list[PunchCandidate],
    attr_priors: dict[str, Any],
    flip_indexes: set[int],
    start_id: int = 1,
) -> list[dict[str, str]]:
    rows = []
    for index, candidate in enumerate(sorted(selected, key=lambda item: item.frame)):
        attrs = estimate_attrs(candidate, attr_priors)
        fighter = swap_fighter(candidate.fighter) if index in flip_indexes else candidate.fighter
        rows.append(
            {
                "id": str(start_id + index),
                "video_id": video["video_id"],
                "agn_index": video["agn_index"],
                "video_key": video["video_key"],
                "frame": str(candidate.frame),
                "fighter": fighter,
                "punch_type": attrs["punch_type"],
                "hand": candidate.hand,
                "target": candidate.target,
                "effectiveness": attrs["effectiveness"],
                "clear": "true",
            }
        )
    return rows


def oracle_matched_rows(
    gt_by_key: dict[str, list[dict[str, str]]],
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, Any],
) -> list[dict[str, str]]:
    rows = build_rows(ready_keys, video_by_key, selected_by_key, attr_priors, {})
    rows_by_key = group_by_key(rows, "video_key")
    id_to_index = {row["id"]: index for index, row in enumerate(rows)}
    for key, pred_rows in rows_by_key.items():
        for match in match_events(gt_by_key.get(key, []), pred_rows):
            row_id = pred_rows[match.pred_index]["id"]
            rows[id_to_index[row_id]]["fighter"] = gt_by_key[key][match.gt_index]["fighter"]
    return rows


def group_by_key(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def fight_group(video: dict[str, str]) -> str:
    return "|".join(
        [
            video["dataset_type"],
            video["data_root"],
            video["fight_index"],
            video["fight_folder"],
        ]
    )


def print_score(label: str, score: dict[str, object], n_rows: int, wins: int) -> None:
    summary = score_summary_with_fighter(score)
    print(
        f"{label}: score={score['macro_score']:.6f},fighter={summary['fighter']:.6f},"
        f"time={summary['time']:.6f},fp={summary['fp_penalty']:.6f},"
        f"wins={wins},n={n_rows}",
        flush=True,
    )


def score_summary_with_fighter(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def print_root_deltas(
    score: dict[str, object],
    baseline: dict[str, object],
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
) -> None:
    by_root: dict[str, list[str]] = defaultdict(list)
    for key in ready_keys:
        by_root[video_by_key[key]["data_root"]].append(key)
    print("root,macro_delta,fighter_delta,n_videos")
    for root, keys in sorted(by_root.items()):
        macro_delta = np.mean(
            [
                score["by_video"][key]["final_score"] - baseline["by_video"][key]["final_score"]
                for key in keys
            ]
        )
        fighter_delta = np.mean(
            [
                score["by_video"][key]["score_fighter"]
                - baseline["by_video"][key]["score_fighter"]
                for key in keys
            ]
        )
        print(f"{root},{macro_delta:.6f},{fighter_delta:.6f},{len(keys)}")


def swap_fighter(fighter: str) -> str:
    return "blue" if fighter == "red" else "red"


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
