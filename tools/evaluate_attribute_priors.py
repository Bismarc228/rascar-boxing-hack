#!/usr/bin/env python3
"""Evaluate fighter and attribute priors on selected pose candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)


LABEL_COLUMNS = ["fighter", "punch_type", "hand", "target", "effectiveness"]
ATTR_COLUMNS = ["punch_type", "hand", "target", "effectiveness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--nms-frames", type=int, default=8)
    parser.add_argument("--nms-group-mode", default="fighter")
    parser.add_argument("--cross-nms-frames", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = [
        path.stem
        for path in sorted(args.tracks_dir.glob("*.jsonl"))
        if path.stem in video_by_key
        and sum(1 for _ in path.open("r", encoding="utf-8")) == int(video_by_key[path.stem]["frame_count"])
    ]
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

    ready_set = set(ready_keys)
    gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    train_gt = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}")

    selected_by_key = {}
    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode=args.nms_group_mode,
        cross_nms_frames=args.cross_nms_frames,
    )
    for key in ready_keys:
        candidates = score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        selected_by_key[key] = select_candidates(candidates, config, None)

    baseline_priors = fit_attr_priors(train_gt)
    prior_tables = build_prior_tables(train_gt, video_by_key)
    baseline_rows = build_rows(
        ready_keys,
        video_by_key,
        selected_by_key,
        baseline_priors,
        prior_tables,
        fighter_mode="candidate",
        attr_mode="pose_target_hand",
        fighter_threshold=0.0,
    )
    baseline_score = score_predictions(gt, baseline_rows)
    print_score("baseline", baseline_score, len(baseline_rows), 0)

    results = []
    fighter_modes = ["candidate", "color_argmax", "root_majority", "root_round_majority"]
    thresholded_modes = ["metadata_if_low_role_conf", "metadata_if_low_color_margin", "color_if_high_margin", "swap_if_low_role_conf"]
    attr_modes = [
        "pose_target_hand",
        "global_majority",
        "root_majority",
        "root_round_majority",
        "root_round_target_hand",
        "metadata_all_root_round",
    ]
    for fighter_mode in fighter_modes:
        for attr_mode in attr_modes:
            add_result(
                results,
                gt,
                baseline_score,
                ready_keys,
                video_by_key,
                selected_by_key,
                baseline_priors,
                prior_tables,
                fighter_mode,
                attr_mode,
                0.0,
            )
    for fighter_mode in thresholded_modes:
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for attr_mode in attr_modes:
                add_result(
                    results,
                    gt,
                    baseline_score,
                    ready_keys,
                    video_by_key,
                    selected_by_key,
                    baseline_priors,
                    prior_tables,
                    fighter_mode,
                    attr_mode,
                    threshold,
                )

    print("score,time,fighter,type,effectiveness,hand,target,wins,n_pred,fighter_mode,attr_mode,threshold")
    for result in sorted(results, reverse=True)[: args.top_k]:
        (
            score,
            time_score,
            fighter_score,
            type_score,
            effectiveness_score,
            hand_score,
            target_score,
            wins,
            n_pred,
            fighter_mode,
            attr_mode,
            threshold,
        ) = result
        print(
            f"{score:.6f},{time_score:.6f},{fighter_score:.6f},{type_score:.6f},"
            f"{effectiveness_score:.6f},{hand_score:.6f},{target_score:.6f},"
            f"{wins},{n_pred},{fighter_mode},{attr_mode},{threshold}"
        )
    return 0


def add_result(
    results: list[tuple[Any, ...]],
    gt: list[dict[str, str]],
    baseline_score: dict[str, Any],
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    baseline_priors: dict[str, Any],
    prior_tables: dict[str, Any],
    fighter_mode: str,
    attr_mode: str,
    fighter_threshold: float,
) -> None:
    rows = build_rows(
        ready_keys,
        video_by_key,
        selected_by_key,
        baseline_priors,
        prior_tables,
        fighter_mode,
        attr_mode,
        fighter_threshold,
    )
    score = score_predictions(gt, rows)
    summary = score_summary(score)
    results.append(
        (
            score["macro_score"],
            summary["time"],
            summary["fighter"],
            summary["punch_type"],
            summary["effectiveness"],
            summary["hand"],
            summary["target"],
            video_wins(score, baseline_score),
            len(rows),
            fighter_mode,
            attr_mode,
            fighter_threshold,
        )
    )


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    baseline_priors: dict[str, Any],
    prior_tables: dict[str, Any],
    fighter_mode: str,
    attr_mode: str,
    fighter_threshold: float,
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        for candidate in sorted(selected_by_key[key], key=lambda item: item.frame):
            pose_attrs = estimate_attrs(candidate, baseline_priors)
            metadata_attrs = predict_attrs(video, candidate, prior_tables, attr_mode, pose_attrs)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": key,
                    "frame": str(candidate.frame),
                    "fighter": predict_fighter(video, candidate, prior_tables, fighter_mode, fighter_threshold),
                    "punch_type": metadata_attrs["punch_type"],
                    "hand": metadata_attrs["hand"],
                    "target": metadata_attrs["target"],
                    "effectiveness": metadata_attrs["effectiveness"],
                    "clear": "true",
                }
            )
            row_id += 1
    return rows


def build_prior_tables(
    train_gt: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> dict[str, Any]:
    tables: dict[str, Any] = {
        "global": majority(train_gt),
        "root": {},
        "root_round": {},
        "root_round_target_hand": {},
    }
    grouped_root: dict[tuple[str], list[dict[str, str]]] = defaultdict(list)
    grouped_root_round: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    grouped_rrth: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in train_gt:
        video = video_by_key[row["video_key"]]
        root_key = (video["data_root"],)
        round_key = (video["data_root"], video["round_number"])
        target_hand_key = (video["data_root"], video["round_number"], row["target"], row["hand"])
        grouped_root[root_key].append(row)
        grouped_root_round[round_key].append(row)
        grouped_rrth[target_hand_key].append(row)
    tables["root"] = {key: majority(rows) for key, rows in grouped_root.items()}
    tables["root_round"] = {key: majority(rows) for key, rows in grouped_root_round.items()}
    tables["root_round_target_hand"] = {key: majority(rows) for key, rows in grouped_rrth.items()}
    return tables


def majority(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {
            "fighter": "blue",
            "punch_type": "hook",
            "hand": "left",
            "target": "head",
            "effectiveness": "landed",
        }
    return {column: Counter(row[column] for row in rows).most_common(1)[0][0] for column in LABEL_COLUMNS}


def predict_attrs(
    video: dict[str, str],
    candidate: PunchCandidate,
    priors: dict[str, Any],
    mode: str,
    pose_attrs: dict[str, str],
) -> dict[str, str]:
    pose_values = {
        "punch_type": pose_attrs["punch_type"],
        "hand": candidate.hand,
        "target": candidate.target,
        "effectiveness": pose_attrs["effectiveness"],
    }
    if mode == "pose_target_hand":
        return pose_values

    root_key = (video["data_root"],)
    round_key = (video["data_root"], video["round_number"])
    target_hand_key = (video["data_root"], video["round_number"], candidate.target, candidate.hand)
    global_prior = priors["global"]
    root_prior = priors["root"].get(root_key, global_prior)
    round_prior = priors["root_round"].get(round_key, root_prior)
    rrth_prior = priors["root_round_target_hand"].get(target_hand_key, round_prior)

    if mode == "global_majority":
        prior = global_prior
    elif mode == "root_majority":
        prior = root_prior
    elif mode == "root_round_majority":
        prior = round_prior
    elif mode == "root_round_target_hand":
        prior = rrth_prior
    elif mode == "metadata_all_root_round":
        return {column: round_prior[column] for column in ATTR_COLUMNS}
    else:
        raise ValueError(f"Unknown attr_mode: {mode}")

    return {
        "punch_type": prior["punch_type"],
        "hand": candidate.hand,
        "target": candidate.target,
        "effectiveness": prior["effectiveness"],
    }


def predict_fighter(
    video: dict[str, str],
    candidate: PunchCandidate,
    priors: dict[str, Any],
    mode: str,
    threshold: float,
) -> str:
    root_key = (video["data_root"],)
    round_key = (video["data_root"], video["round_number"])
    root_fighter = priors["root"].get(root_key, priors["global"])["fighter"]
    round_fighter = priors["root_round"].get(round_key, priors["root"].get(root_key, priors["global"]))["fighter"]
    color_fighter = "red" if candidate.features.get("attacker_red_score", 0.0) >= candidate.features.get("attacker_blue_score", 0.0) else "blue"
    margin = abs(candidate.features.get("attacker_color_margin", 0.0))
    role_conf = candidate.features.get("role_conf", 0.0)

    if mode == "candidate":
        return candidate.fighter
    if mode == "color_argmax":
        return color_fighter
    if mode == "root_majority":
        return root_fighter
    if mode == "root_round_majority":
        return round_fighter
    if mode == "metadata_if_low_role_conf":
        return round_fighter if role_conf < threshold else candidate.fighter
    if mode == "metadata_if_low_color_margin":
        return round_fighter if margin < threshold else candidate.fighter
    if mode == "color_if_high_margin":
        return color_fighter if margin >= threshold else candidate.fighter
    if mode == "swap_if_low_role_conf":
        return ("blue" if candidate.fighter == "red" else "red") if role_conf < threshold else candidate.fighter
    raise ValueError(f"Unknown fighter_mode: {mode}")


def print_score(label: str, score: dict[str, Any], n_pred: int, wins: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},time={summary['time']:.6f},"
        f"fighter={summary['fighter']:.6f},type={summary['punch_type']:.6f},"
        f"effectiveness={summary['effectiveness']:.6f},hand={summary['hand']:.6f},"
        f"target={summary['target']:.6f},wins={wins},n_pred={n_pred}"
    )


def score_summary(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "punch_type": float(np.mean([item["score_punch_type"] for item in by_video.values()])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
    }


def video_wins(score: dict[str, Any], baseline: dict[str, Any], eps: float = 1e-9) -> int:
    return sum(
        int(score["by_video"][key]["final_score"] > baseline["by_video"][key]["final_score"] + eps)
        for key in baseline["by_video"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
