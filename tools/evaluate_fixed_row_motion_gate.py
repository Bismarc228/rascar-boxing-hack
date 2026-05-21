#!/usr/bin/env python3
"""Evaluate crop-motion features as a fixed-row keep/drop gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_crop_motion_rerank import compute_motion, load_records
from tools.evaluate_exchange_state_gate import (
    apply_keep_mask,
    build_labels,
    fight_group,
    fit_oof_probabilities,
    index_candidates,
    parse_floats,
    parse_ints,
    print_label_summary,
    print_score,
    row_features,
    score_summary,
)
from tools.evaluate_pose_selection_variants import video_wins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--feature-windows", default="4,8,16")
    parser.add_argument("--motion-offsets", default="-4,-2,0,2,4")
    parser.add_argument("--resize-width", type=int, default=240)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--thresholds", default="0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.35,0.4,0.5")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-oof-rows", type=Path)
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
            apply_temporal_context(
                score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config),
                config,
            )
        )
        for key in keys
    }
    offsets = parse_ints(args.motion_offsets)
    motion_by_key = compute_all_motion(args, pred_rows, video_by_key, offsets)
    windows = parse_ints(args.feature_windows)
    x = np.stack(
        [
            combined_features(
                row,
                video_by_key[row["video_key"]],
                indexed_by_key[row["video_key"]],
                motion_by_key[row["video_key"]],
                offsets,
                args.match_window,
                windows,
            )
            for row in pred_rows
        ]
    )
    y, labels = build_labels(pred_rows, gt_rows)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    p_keep = fit_oof_probabilities(args.model, x, y, groups)

    oracle_fp_rows = apply_keep_mask(pred_rows, np.asarray([label != "fp" for label in labels], dtype=bool))
    oracle_fp_score = score_predictions(gt_rows, oracle_fp_rows)
    print_score("oracle_drop_unmatched_fp", oracle_fp_score, video_wins(oracle_fp_score, baseline), len(oracle_fp_rows))
    print_label_summary(labels, p_keep)

    results = []
    for threshold in parse_floats(args.thresholds):
        keep_mask = p_keep >= threshold
        rows = apply_keep_mask(pred_rows, keep_mask)
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        results.append(
            {
                "score": score["macro_score"],
                "delta": score["macro_score"] - baseline["macro_score"],
                "fighter": summary["fighter"],
                "time": summary["time"],
                "fp": summary["fp"],
                "wins": video_wins(score, baseline),
                "n_kept": len(rows),
                "n_dropped": int((~keep_mask).sum()),
                "threshold": threshold,
            }
        )
    print("score,delta,fighter,time,fp,wins,n_kept,n_dropped,threshold")
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print(
            f"{item['score']:.6f},{item['delta']:.6f},{item['fighter']:.6f},"
            f"{item['time']:.6f},{item['fp']:.6f},{item['wins']},"
            f"{item['n_kept']},{item['n_dropped']},{item['threshold']}"
        )
    if args.write_oof_rows:
        threshold = args.write_threshold
        if threshold is None:
            threshold = max(results, key=lambda row: row["score"])["threshold"]
        rows = apply_keep_mask(pred_rows, p_keep >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} threshold={threshold} n_rows={len(rows)}")
    return 0


def compute_all_motion(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    offsets: list[int],
) -> dict[str, dict[int, tuple[float, float, float]]]:
    wanted_by_key: dict[str, set[int]] = {}
    for row in pred_rows:
        key = row["video_key"]
        frame = as_int(row["frame"], "frame")
        frame_count = as_int(video_by_key[key]["frame_count"], "frame_count")
        for offset in offsets:
            wanted_by_key.setdefault(key, set()).add(max(0, min(frame_count - 1, frame + offset)))
    tasks = [
        (
            key,
            args.data_root / video_by_key[key]["video_path"],
            args.tracks_dir / f"{key}.jsonl",
            wanted_by_key[key],
            args.resize_width,
            args.quiet,
        )
        for key in sorted(wanted_by_key)
    ]
    if args.jobs <= 1:
        return {key: motion for key, motion in map(compute_motion_task, tasks)}
    output: dict[str, dict[int, tuple[float, float, float]]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(compute_motion_task, task) for task in tasks]
        for future in as_completed(futures):
            key, motion = future.result()
            output[key] = motion
            print(f"motion_done={key} frames={len(motion)}", flush=True)
    return output


def compute_motion_task(
    task: tuple[str, Path, Path, set[int], int, bool],
) -> tuple[str, dict[int, tuple[float, float, float]]]:
    key, video_path, tracks_path, wanted_frames, resize_width, quiet = task
    records = load_records(tracks_path, wanted_frames)
    return key, compute_motion(video_path, records, resize_width, quiet)


def combined_features(
    row: dict[str, str],
    video: dict[str, str],
    candidates_by_frame,
    motion_by_frame: dict[int, tuple[float, float, float]],
    offsets: list[int],
    match_window: int,
    windows: list[int],
) -> np.ndarray:
    base = row_features(row, video, candidates_by_frame, match_window, windows).tolist()
    frame = as_int(row["frame"], "frame")
    frame_count = as_int(video["frame_count"], "frame_count")
    values = []
    for offset in offsets:
        current = max(0, min(frame_count - 1, frame + offset))
        crop_motion, global_motion, ratio = motion_by_frame.get(current, (0.0, 0.0, 0.0))
        values.extend(
            [
                crop_motion / 255.0,
                global_motion / 255.0,
                min(ratio, 10.0) / 10.0,
                float(np.log1p(crop_motion)),
                float(np.log1p(global_motion)),
                float(np.log1p(ratio)),
            ]
        )
    return np.asarray(base + values, dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
