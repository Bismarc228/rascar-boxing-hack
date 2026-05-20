#!/usr/bin/env python3
"""Evaluate fighter-label smoothing while keeping selected pose events fixed."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    estimate_attrs,
    estimate_count,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)


FIGHTERS = {"red", "blue"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("data/processed/pose_tracks/val_yolo11s_conf035"),
    )
    parser.add_argument("--threshold", type=float, default=1.15)
    parser.add_argument("--nms-frames", type=int, default=10)
    parser.add_argument("--nms-group-mode", default="fighter_hand")
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--count-mode", default="root_count")
    parser.add_argument("--count-multiplier", type=float, default=1.0)
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
    val_videos = [video_by_key[key] for key in ready_keys]
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    gt = [row for row in punches if row.get("clear") == "true" and row["video_key"] in ready_set]
    train_gt = [
        row for row in punches if row.get("clear") == "true" and row["video_key"] not in ready_set
    ]
    attr_priors = fit_attr_priors(train_gt)

    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode=args.nms_group_mode,
        cross_nms_frames=args.cross_nms_frames,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
    )
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}")
    print(
        "config="
        f"tracks_dir={args.tracks_dir},threshold={args.threshold},"
        f"nms={args.nms_frames},group={args.nms_group_mode},"
        f"cross={args.cross_nms_frames},count={args.count_mode},"
        f"multiplier={args.count_multiplier}"
    )

    selected_by_key: dict[str, list[PunchCandidate]] = {}
    track_info_by_key: dict[str, TrackInfo] = {}
    for key in ready_keys:
        tracks_path = args.tracks_dir / f"{key}.jsonl"
        candidates = score_pose_tracks(tracks_path, replace(config, min_score=0.0))
        count = estimate_count(config, video_by_key[key], train_videos, train_gt, [])
        selected_by_key[key] = select_candidates(candidates, config, count)
        track_info_by_key[key] = load_track_info(tracks_path)

    baseline_rows = build_rows(
        ready_keys,
        video_by_key,
        selected_by_key,
        attr_priors,
        lambda key, candidate: candidate.fighter,
    )
    baseline_score = score_predictions(gt, baseline_rows)

    variants = [
        ("baseline", baseline_rows),
        (
            "full_video_swap",
            build_rows(
                ready_keys,
                video_by_key,
                selected_by_key,
                attr_priors,
                lambda key, candidate: swap_fighter(candidate.fighter),
            ),
        ),
        (
            "track_role_majority",
            build_rows(
                ready_keys,
                video_by_key,
                selected_by_key,
                attr_priors,
                lambda key, candidate: smooth_by_track_role(track_info_by_key[key], candidate),
            ),
        ),
        (
            "track_color_mean",
            build_rows(
                ready_keys,
                video_by_key,
                selected_by_key,
                attr_priors,
                lambda key, candidate: smooth_by_track_color(track_info_by_key[key], candidate),
            ),
        ),
    ]
    variants.append(("oracle_matched_fighter", oracle_matched_fighter_rows(gt, baseline_rows)))

    print("variant,macro,delta,fighter,time,fp,wins,n_pred")
    scores: dict[str, dict[str, Any]] = {}
    for label, rows in variants:
        score = score_predictions(gt, rows)
        scores[label] = score
        summary = score_summary(score)
        print(
            f"{label},{score['macro_score']:.6f},"
            f"{score['macro_score'] - baseline_score['macro_score']:.6f},"
            f"{summary['fighter']:.6f},{summary['time']:.6f},"
            f"{summary['fp_penalty']:.6f},{video_wins(score, baseline_score)},"
            f"{len(rows)}"
        )

    print_root_deltas(variants, scores, baseline_score, video_by_key)
    print_video_deltas(variants, scores, baseline_score, video_by_key)
    return 0


class TrackInfo:
    def __init__(
        self,
        frame_role_track: dict[tuple[int, str], int],
        role_majority_by_track: dict[int, str],
        color_fighter_by_track: dict[int, str],
    ) -> None:
        self.frame_role_track = frame_role_track
        self.role_majority_by_track = role_majority_by_track
        self.color_fighter_by_track = color_fighter_by_track


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


def load_track_info(tracks_path: Path) -> TrackInfo:
    frame_role_track: dict[tuple[int, str], int] = {}
    role_counts: dict[int, Counter[str]] = defaultdict(Counter)
    score_sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    with tracks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame"])
            for role, fighter in record.get("fighters", {}).items():
                if role not in FIGHTERS or "track_id" not in fighter:
                    continue
                track_id = int(fighter["track_id"])
                frame_role_track[(frame, role)] = track_id
                role_counts[track_id][role] += 1
                score_sums[track_id][0] += float(fighter.get("score_red", 0.0))
                score_sums[track_id][1] += float(fighter.get("score_blue", 0.0))
                score_sums[track_id][2] += 1.0

    role_majority = {
        track_id: counts.most_common(1)[0][0] for track_id, counts in role_counts.items()
    }
    color_fighter = {}
    for track_id, (red_sum, blue_sum, count) in score_sums.items():
        red_mean = red_sum / max(1.0, count)
        blue_mean = blue_sum / max(1.0, count)
        color_fighter[track_id] = "red" if red_mean >= blue_mean else "blue"
    return TrackInfo(frame_role_track, role_majority, color_fighter)


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, Any],
    fighter_fn: Callable[[str, PunchCandidate], str],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        for candidate in sorted(selected_by_key[key], key=lambda item: item.frame):
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": key,
                    "frame": str(candidate.frame),
                    "fighter": fighter_fn(key, candidate),
                    "punch_type": attrs["punch_type"],
                    "hand": candidate.hand,
                    "target": candidate.target,
                    "effectiveness": attrs["effectiveness"],
                    "clear": "true",
                }
            )
            row_id += 1
    return rows


def smooth_by_track_role(track_info: TrackInfo, candidate: PunchCandidate) -> str:
    track_id = track_info.frame_role_track.get((candidate.frame, candidate.fighter))
    if track_id is None:
        return candidate.fighter
    return track_info.role_majority_by_track.get(track_id, candidate.fighter)


def smooth_by_track_color(track_info: TrackInfo, candidate: PunchCandidate) -> str:
    track_id = track_info.frame_role_track.get((candidate.frame, candidate.fighter))
    if track_id is None:
        return candidate.fighter
    return track_info.color_fighter_by_track.get(track_id, candidate.fighter)


def oracle_matched_fighter_rows(
    gt_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    gt_by_video = group_by_key(gt_rows, "video_key")
    pred_by_video = group_by_key(baseline_rows, "video_key")
    output = [dict(row) for row in baseline_rows]
    global_index_by_id = {row["id"]: index for index, row in enumerate(output)}

    for key, pred_rows in pred_by_video.items():
        gt_video_rows = gt_by_video.get(key, [])
        for match in match_events(gt_video_rows, pred_rows):
            pred_id = pred_rows[match.pred_index]["id"]
            output[global_index_by_id[pred_id]]["fighter"] = gt_video_rows[match.gt_index]["fighter"]
    return output


def group_by_key(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def score_summary(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def video_wins(score: dict[str, Any], baseline: dict[str, Any], eps: float = 1e-9) -> int:
    return sum(
        int(score["by_video"][key]["final_score"] > baseline["by_video"][key]["final_score"] + eps)
        for key in baseline["by_video"]
    )


def print_root_deltas(
    variants: list[tuple[str, list[dict[str, str]]]],
    scores: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    video_by_key: dict[str, dict[str, str]],
) -> None:
    root_keys = sorted({video_by_key[key]["data_root"] for key in baseline["by_video"]})
    print("root,variant,macro_delta,fighter_delta,time_delta,fp_delta,n_videos")
    for root in root_keys:
        keys = [
            key
            for key in baseline["by_video"]
            if video_by_key[key]["data_root"] == root
        ]
        for label, _rows in variants:
            score = scores[label]
            print(
                f"{root},{label},"
                f"{mean_delta(score, baseline, keys, 'final_score'):.6f},"
                f"{mean_delta(score, baseline, keys, 'score_fighter'):.6f},"
                f"{mean_delta(score, baseline, keys, 'score_time'):.6f},"
                f"{mean_delta(score, baseline, keys, 'fp_penalty'):.6f},"
                f"{len(keys)}"
            )


def print_video_deltas(
    variants: list[tuple[str, list[dict[str, str]]]],
    scores: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    video_by_key: dict[str, dict[str, str]],
) -> None:
    print("video_key,root,variant,macro_delta,fighter_delta,time_delta,fp_delta,n_pred")
    for key in sorted(baseline["by_video"]):
        root = video_by_key[key]["data_root"]
        for label, _rows in variants:
            item = scores[label]["by_video"][key]
            base = baseline["by_video"][key]
            print(
                f"{key},{root},{label},"
                f"{item['final_score'] - base['final_score']:.6f},"
                f"{item['score_fighter'] - base['score_fighter']:.6f},"
                f"{item['score_time'] - base['score_time']:.6f},"
                f"{item['fp_penalty'] - base['fp_penalty']:.6f},"
                f"{item['n_pred']}"
            )


def mean_delta(
    score: dict[str, Any],
    baseline: dict[str, Any],
    keys: list[str],
    metric: str,
) -> float:
    return float(
        np.mean(
            [
                score["by_video"][key][metric] - baseline["by_video"][key][metric]
                for key in keys
            ]
        )
    )


def swap_fighter(fighter: str) -> str:
    return "blue" if fighter == "red" else "red"


if __name__ == "__main__":
    raise SystemExit(main())
