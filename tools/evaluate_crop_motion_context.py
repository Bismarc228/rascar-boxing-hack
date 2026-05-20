#!/usr/bin/env python3
"""Evaluate crop-motion rescoring on top of current pose-context candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
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
    apply_temporal_context,
    estimate_attrs,
    estimate_count,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from tools.evaluate_crop_motion_rerank import compute_motion, load_records
from tools.evaluate_pose_selection_variants import score_summary, video_wins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=240)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--nms-frames", type=int, default=8)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--nms-group-mode", default="fighter_hand")
    parser.add_argument("--count-mode", default="threshold")
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    parser.add_argument("--context-feature", default="same_count")
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--motion-alphas", default="-0.05,-0.03,0.0,0.03,0.05,0.1")
    parser.add_argument("--motion-betas", default="-0.05,0.0,0.05,0.1")
    parser.add_argument("--video-keys", default="")
    parser.add_argument("--max-candidates-per-video", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if args.video_keys:
        requested = {value for value in args.video_keys.split(",") if value}
        ready_keys = [key for key in ready_keys if key in requested]
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

    ready_set = set(ready_keys)
    gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    train_gt = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    attr_priors = fit_attr_priors(train_gt)
    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode=args.nms_group_mode,
        cross_nms_frames=args.cross_nms_frames,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}")
    print(
        "config="
        f"threshold={args.threshold},same={args.nms_frames},cross={args.cross_nms_frames},"
        f"group={args.nms_group_mode},count={args.count_mode}:{args.count_multiplier},"
        f"context={args.context_feature}:{args.context_window}:{args.context_alpha},"
        f"resize_width={args.resize_width}"
    )

    candidates_by_key: dict[str, list[PunchCandidate]] = {}
    counts_by_key: dict[str, int | None] = {}
    for key in progress(ready_keys, "motion-context", args.quiet):
        video = video_by_key[key]
        raw_candidates = score_pose_tracks(
            args.tracks_dir / f"{key}.jsonl",
            PoseHeuristicConfig(min_score=0.0),
        )
        candidates_by_key[key] = apply_temporal_context(raw_candidates, config)
        if args.max_candidates_per_video is not None:
            candidates_by_key[key] = sorted(
                candidates_by_key[key],
                key=lambda candidate: candidate.score,
                reverse=True,
            )[: args.max_candidates_per_video]
        counts_by_key[key] = estimate_count(config, video, train_videos, train_gt, [])

    motion_by_key = compute_all_motion(
        args.data_root,
        args.tracks_dir,
        ready_keys,
        video_by_key,
        candidates_by_key,
        args.resize_width,
        args.quiet,
        args.jobs,
    )

    baseline_rows = build_rows(
        ready_keys,
        video_by_key,
        candidates_by_key,
        counts_by_key,
        attr_priors,
        config,
    )
    baseline_score = score_predictions(gt, baseline_rows)
    baseline_summary = score_summary_with_fighter(baseline_score)
    print(
        "baseline,"
        f"{baseline_score['macro_score']:.6f},{baseline_summary['time']:.6f},"
        f"{baseline_summary['fighter']:.6f},{baseline_summary['fp_penalty']:.6f},"
        f"{len(baseline_rows)}"
    )

    results = []
    for alpha in parse_floats(args.motion_alphas):
        for beta in parse_floats(args.motion_betas):
            rescored_by_key = {
                key: [
                    rescore_with_motion(
                        candidate,
                        motion_by_key[key].get(candidate.frame, (0.0, 0.0, 0.0)),
                        alpha,
                        beta,
                    )
                    for candidate in candidates_by_key[key]
                ]
                for key in ready_keys
            }
            rows = build_rows(
                ready_keys,
                video_by_key,
                rescored_by_key,
                counts_by_key,
                attr_priors,
                config,
            )
            score = score_predictions(gt, rows)
            summary = score_summary_with_fighter(score)
            results.append(
                {
                    "score": score,
                    "macro": score["macro_score"],
                    "time": summary["time"],
                    "fighter": summary["fighter"],
                    "fp_penalty": summary["fp_penalty"],
                    "wins": video_wins(score, baseline_score),
                    "n_pred": len(rows),
                    "alpha": alpha,
                    "beta": beta,
                    "changed": count_changed(baseline_rows, rows),
                }
            )

    print("score,delta,time,fighter,fp_penalty,wins,n_pred,changed,motion_alpha,motion_beta")
    for result in sorted(results, key=lambda item: item["macro"], reverse=True)[: args.top_k]:
        print(
            f"{result['macro']:.6f},{result['macro'] - baseline_score['macro_score']:.6f},"
            f"{result['time']:.6f},{result['fighter']:.6f},{result['fp_penalty']:.6f},"
            f"{result['wins']},{result['n_pred']},{result['changed']},"
            f"{result['alpha']},{result['beta']}"
        )

    print_root_deltas(results, baseline_score, ready_keys, video_by_key, args.top_k)
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


def compute_all_motion(
    data_root: Path,
    tracks_dir: Path,
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    resize_width: int,
    quiet: bool,
    jobs: int,
) -> dict[str, dict[int, tuple[float, float, float]]]:
    tasks = [
        (
            key,
            data_root / video_by_key[key]["video_path"],
            tracks_dir / f"{key}.jsonl",
            sorted({candidate.frame for candidate in candidates_by_key[key]}),
            resize_width,
            quiet,
        )
        for key in ready_keys
    ]
    if jobs <= 1:
        return {key: motion for key, motion in map(compute_motion_task, tasks)}

    output: dict[str, dict[int, tuple[float, float, float]]] = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(compute_motion_task, task) for task in tasks]
        for future in as_completed(futures):
            key, motion = future.result()
            output[key] = motion
    return output


def compute_motion_task(
    task: tuple[str, Path, Path, list[int], int, bool],
) -> tuple[str, dict[int, tuple[float, float, float]]]:
    key, video_path, tracks_path, wanted_frames, resize_width, quiet = task
    records = load_records(tracks_path, set(wanted_frames))
    return key, compute_motion(video_path, records, resize_width, quiet)


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    counts_by_key: dict[str, int | None],
    attr_priors: dict[str, Any],
    config: PoseHeuristicConfig,
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        selected = select_candidates(candidates_by_key[key], config, counts_by_key[key])
        for candidate in selected:
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


def rescore_with_motion(
    candidate: PunchCandidate,
    motion: tuple[float, float, float],
    alpha: float,
    beta: float,
) -> PunchCandidate:
    if alpha == 0.0 and beta == 0.0:
        return candidate
    crop_motion, _global_motion, ratio = motion
    score = candidate.score * max(
        0.01,
        1.0 + alpha * np.log1p(crop_motion) + beta * np.log1p(ratio),
    )
    features = dict(candidate.features)
    features["crop_motion"] = float(crop_motion)
    features["motion_ratio"] = float(ratio)
    return PunchCandidate(
        candidate.video_key,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
        float(score),
        features,
    )


def score_summary_with_fighter(score: dict[str, object]) -> dict[str, float]:
    summary = score_summary(score)
    summary["fighter"] = float(np.mean([item["score_fighter"] for item in score["by_video"].values()]))
    return summary


def count_changed(left: list[dict[str, str]], right: list[dict[str, str]]) -> int:
    return sum(
        1
        for a, b in zip(left, right)
        if a["video_key"] != b["video_key"]
        or a["frame"] != b["frame"]
        or a["fighter"] != b["fighter"]
        or a["hand"] != b["hand"]
    ) + abs(len(left) - len(right))


def print_root_deltas(
    results: list[dict[str, object]],
    baseline: dict[str, object],
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    top_k: int,
) -> None:
    by_root: dict[str, list[str]] = defaultdict(list)
    for key in ready_keys:
        by_root[video_by_key[key]["data_root"]].append(key)
    print("root,rank,motion_alpha,motion_beta,macro_delta,time_delta,fp_delta,n_videos")
    for rank, result in enumerate(
        sorted(results, key=lambda item: item["macro"], reverse=True)[: max(5, min(top_k, 10))],
        start=1,
    ):
        score = result["score"]
        for root, keys in sorted(by_root.items()):
            macro_delta = np.mean(
                [
                    score["by_video"][key]["final_score"] - baseline["by_video"][key]["final_score"]
                    for key in keys
                ]
            )
            time_delta = np.mean(
                [
                    score["by_video"][key]["score_time"] - baseline["by_video"][key]["score_time"]
                    for key in keys
                ]
            )
            fp_delta = np.mean(
                [
                    score["by_video"][key]["fp_penalty"] - baseline["by_video"][key]["fp_penalty"]
                    for key in keys
                ]
            )
            print(
                f"{root},{rank},{result['alpha']},{result['beta']},"
                f"{macro_delta:.6f},{time_delta:.6f},{fp_delta:.6f},{len(keys)}"
            )


def progress(items: list[str], desc: str, quiet: bool):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc, leave=False, disable=quiet, mininterval=5.0)
    except Exception:
        if not quiet:
            print(f"{desc}: {len(items)} items")
        return items


def parse_floats(values: str) -> list[float]:
    return [float(value) for value in values.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
