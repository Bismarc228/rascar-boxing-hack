#!/usr/bin/env python3
"""Evaluate snapping primary pose timings to secondary pose candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from tools.evaluate_pose_model_agreement import complete_track_keys
from tools.evaluate_pose_selection_variants import (
    estimate_count,
    score_summary,
    video_wins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--primary-tracks-dir",
        type=Path,
        default=Path("data/processed/pose_tracks/val_yolo11s_conf035"),
    )
    parser.add_argument(
        "--secondary-tracks-dir",
        type=Path,
        default=Path("data/processed/pose_tracks/val_yolo26m_conf035"),
    )
    parser.add_argument("--primary-threshold", type=float, default=1.15)
    parser.add_argument("--primary-nms-frames", type=int, default=10)
    parser.add_argument("--primary-nms-group-mode", default="fighter_hand")
    parser.add_argument("--primary-cross-nms-frames", type=int, default=4)
    parser.add_argument("--primary-count-mode", default="root_count")
    parser.add_argument("--primary-count-multiplier", type=float, default=1.0)
    parser.add_argument("--secondary-kp-conf", type=float, default=0.35)
    parser.add_argument("--secondary-velocity-lag", type=int, default=3)
    parser.add_argument("--secondary-nms-frames", type=int, default=2)
    parser.add_argument("--secondary-cross-nms-frames", type=int, default=0)
    parser.add_argument("--secondary-context-feature", default="none")
    parser.add_argument("--secondary-context-window", type=int, default=4)
    parser.add_argument("--secondary-context-alpha", type=float, default=0.0)
    parser.add_argument("--windows", default="2,4,6,8")
    parser.add_argument("--secondary-thresholds", default="0.65,0.85,1.05")
    parser.add_argument("--choose", default="nearest,score_dist")
    parser.add_argument("--modes", default="replace,midpoint,weighted")
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    primary_keys = set(complete_track_keys(args.primary_tracks_dir, video_by_key))
    secondary_keys = set(complete_track_keys(args.secondary_tracks_dir, video_by_key))
    ready_keys = sorted(primary_keys & secondary_keys)
    if not ready_keys:
        print("No common complete validation tracks found")
        return 2

    ready_set = set(ready_keys)
    gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    train_gt = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    train_counts = Counter(row["video_key"] for row in train_gt)
    gt_counts = Counter(row["video_key"] for row in gt)
    attr_priors = fit_attr_priors(train_gt)
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}", flush=True)

    primary_raw = {
        key: score_pose_tracks(args.primary_tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        for key in ready_keys
    }
    secondary_raw = {
        key: apply_temporal_context(
            score_pose_tracks(
                args.secondary_tracks_dir / f"{key}.jsonl",
                PoseHeuristicConfig(
                    kp_conf=args.secondary_kp_conf,
                    velocity_lag=args.secondary_velocity_lag,
                    min_score=0.0,
                ),
            ),
            PoseHeuristicConfig(
                context_feature=args.secondary_context_feature,
                context_window=args.secondary_context_window,
                context_alpha=args.secondary_context_alpha,
            ),
        )
        for key in ready_keys
    }

    selected_primary = {
        key: select_primary(
            key,
            video_by_key,
            primary_raw[key],
            train_videos,
            train_counts,
            gt_counts,
            args,
        )
        for key in ready_keys
    }
    baseline_rows = build_rows(ready_keys, video_by_key, selected_primary, attr_priors)
    baseline_score = score_predictions(gt, baseline_rows)
    print_score("primary_ref", baseline_score, len(baseline_rows), 0)

    windows = parse_ints(args.windows)
    secondary_thresholds = parse_floats(args.secondary_thresholds)
    choose_modes = [value for value in args.choose.split(",") if value]
    snap_modes = [value for value in args.modes.split(",") if value]
    secondary_context = (
        f"{args.secondary_context_feature}:"
        f"{args.secondary_context_window}:"
        f"{args.secondary_context_alpha}"
    )

    results = []
    for sec_thr in secondary_thresholds:
        secondary_pool = {
            key: select_candidates(
                secondary_raw[key],
                PoseHeuristicConfig(
                    min_score=sec_thr,
                    nms_frames=args.secondary_nms_frames,
                    nms_group_mode="fighter_hand",
                    cross_nms_frames=args.secondary_cross_nms_frames,
                ),
                None,
            )
            for key in ready_keys
        }
        for window in windows:
            for choose in choose_modes:
                for mode in snap_modes:
                    snapped_by_key, changed, root_shifts = snap_all(
                        ready_keys,
                        video_by_key,
                        selected_primary,
                        secondary_pool,
                        window,
                        choose,
                        mode,
                    )
                    rows = build_rows(ready_keys, video_by_key, snapped_by_key, attr_priors)
                    score = score_predictions(gt, rows)
                    summary = score_summary(score)
                    results.append(
                        (
                            score["macro_score"],
                            summary["time"],
                            summary["fp_penalty"],
                            video_wins(score, baseline_score),
                            changed,
                            len(rows),
                            secondary_context,
                            window,
                            sec_thr,
                            choose,
                            mode,
                            format_root_score_deltas(video_by_key, ready_keys, baseline_score, score),
                            format_root_shifts(root_shifts),
                        )
                    )

    print(
        "score,time,fp,wins,changed,n_pred,secondary_context,window,sec_thr,"
        "choose,mode,root_score_deltas,root_shift_summary"
    )
    for result in sorted(results, reverse=True)[: args.top_k]:
        print(
            f"{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},{result[3]},"
            f"{result[4]},{result[5]},{result[6]},{result[7]},{result[8]},"
            f"{result[9]},{result[10]},{result[11]},{result[12]}"
        )
    return 0


def select_primary(
    key: str,
    video_by_key: dict[str, dict[str, str]],
    candidates: list[PunchCandidate],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    args: argparse.Namespace,
) -> list[PunchCandidate]:
    video = video_by_key[key]
    count = estimate_count(
        video,
        train_videos,
        train_counts,
        gt_counts,
        args.primary_count_mode,
        args.primary_count_multiplier,
    )
    return select_candidates(
        candidates,
        PoseHeuristicConfig(
            min_score=args.primary_threshold,
            nms_frames=args.primary_nms_frames,
            nms_group_mode=args.primary_nms_group_mode,
            cross_nms_frames=args.primary_cross_nms_frames,
        ),
        count,
    )


def snap_all(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    primary_by_key: dict[str, list[PunchCandidate]],
    secondary_by_key: dict[str, list[PunchCandidate]],
    window: int,
    choose: str,
    mode: str,
) -> tuple[dict[str, list[PunchCandidate]], int, dict[str, list[int]]]:
    snapped_by_key: dict[str, list[PunchCandidate]] = {}
    changed = 0
    root_deltas: dict[str, list[int]] = {}
    for key in ready_keys:
        root = video_by_key[key]["data_root"]
        root_deltas.setdefault(root, [])
        snapped = []
        for candidate in primary_by_key[key]:
            match = choose_match(candidate, secondary_by_key[key], window, choose)
            if match is None:
                snapped.append(candidate)
                continue
            frame = snap_frame(candidate, match, mode)
            frame = max(0, min(int(video_by_key[key]["frame_count"]) - 1, frame))
            delta = frame - candidate.frame
            root_deltas[root].append(delta)
            if delta != 0:
                changed += 1
            features = dict(candidate.features)
            features["snap_delta"] = float(delta)
            features["snap_secondary_score"] = float(match.score)
            snapped.append(
                PunchCandidate(
                    candidate.video_key,
                    frame,
                    candidate.fighter,
                    candidate.hand,
                    candidate.target,
                    candidate.score,
                    features,
                )
            )
        snapped_by_key[key] = sorted(snapped, key=lambda item: item.frame)
    return snapped_by_key, changed, root_deltas


def choose_match(
    candidate: PunchCandidate,
    pool: list[PunchCandidate],
    window: int,
    choose: str,
) -> PunchCandidate | None:
    matches = [
        item
        for item in pool
        if item.fighter == candidate.fighter
        and item.hand == candidate.hand
        and abs(item.frame - candidate.frame) <= window
    ]
    if not matches:
        return None
    if choose == "nearest":
        return min(matches, key=lambda item: (abs(item.frame - candidate.frame), -item.score))
    if choose == "score_dist":
        return max(matches, key=lambda item: (item.score / (1.0 + abs(item.frame - candidate.frame)), item.score))
    raise ValueError(f"Unknown choose mode: {choose}")


def snap_frame(candidate: PunchCandidate, match: PunchCandidate, mode: str) -> int:
    if mode == "replace":
        return match.frame
    if mode == "midpoint":
        return round((candidate.frame + match.frame) * 0.5)
    if mode == "weighted":
        total = max(1e-9, candidate.score + match.score)
        return round((candidate.frame * candidate.score + match.frame * match.score) / total)
    raise ValueError(f"Unknown snap mode: {mode}")


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        for candidate in sorted(candidates_by_key[key], key=lambda item: item.frame):
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": key,
                    "frame": str(max(0, min(int(video["frame_count"]) - 1, candidate.frame))),
                    "fighter": candidate.fighter,
                    "punch_type": attrs["punch_type"],
                    "hand": candidate.hand,
                    "target": candidate.target,
                    "effectiveness": attrs["effectiveness"],
                    "clear": "true",
                }
            )
            row_id += 1
    return rows


def print_score(label: str, score: dict[str, object], n_pred: int, wins: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},time={summary['time']:.6f},"
        f"fp={summary['fp_penalty']:.6f},wins={wins},n_pred={n_pred}",
        flush=True,
    )


def format_root_score_deltas(
    video_by_key: dict[str, dict[str, str]],
    ready_keys: list[str],
    baseline_score: dict[str, object],
    candidate_score: dict[str, object],
) -> str:
    by_root: dict[str, list[str]] = {}
    for key in ready_keys:
        by_root.setdefault(video_by_key[key]["data_root"], []).append(key)
    parts = []
    baseline_by_video = baseline_score["by_video"]
    candidate_by_video = candidate_score["by_video"]
    for root in sorted(by_root):
        keys = by_root[root]
        baseline = sum(baseline_by_video[key]["final_score"] for key in keys) / len(keys)
        candidate = sum(candidate_by_video[key]["final_score"] for key in keys) / len(keys)
        wins = sum(
            candidate_by_video[key]["final_score"] > baseline_by_video[key]["final_score"] + 1e-9
            for key in keys
        )
        parts.append(f"{root}:{candidate - baseline:+.6f}/{wins}-{len(keys) - wins}")
    return ";".join(parts)


def format_root_shifts(root_deltas: dict[str, list[int]]) -> str:
    parts = []
    for root in sorted(root_deltas):
        values = root_deltas[root]
        if values:
            mean = sum(values) / len(values)
            mean_abs = sum(abs(value) for value in values) / len(values)
            parts.append(f"{root}:{mean:.2f}/{mean_abs:.2f}/{len(values)}")
        else:
            parts.append(f"{root}:0.00/0.00/0")
    return ";".join(parts)


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
