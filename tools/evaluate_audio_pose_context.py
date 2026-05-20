#!/usr/bin/env python3
"""Evaluate audio onset rescoring on top of current pose-context candidates."""

from __future__ import annotations

import argparse
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
from tools.evaluate_audio_onsets import audio_onset_by_frame, load_audio
from tools.evaluate_pose_selection_variants import score_summary, video_wins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--nms-frames", type=int, default=8)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--nms-group-mode", default="fighter_hand")
    parser.add_argument("--count-mode", default="threshold")
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    parser.add_argument("--context-feature", default="same_count")
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--audio-windows", default="0,3,6,12")
    parser.add_argument("--audio-alphas", default="-0.25,-0.1,-0.03,0.0,0.03,0.05,0.1,0.25")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if args.max_videos:
        ready_keys = ready_keys[: args.max_videos]
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
        f"context={args.context_feature}:{args.context_window}:{args.context_alpha}"
    )

    candidates_by_key: dict[str, list[PunchCandidate]] = {}
    counts_by_key: dict[str, int | None] = {}
    onsets_by_key: dict[str, np.ndarray] = {}
    for key in progress(ready_keys, "audio-context", args.quiet):
        video = video_by_key[key]
        raw_candidates = score_pose_tracks(
            args.tracks_dir / f"{key}.jsonl",
            PoseHeuristicConfig(min_score=0.0),
        )
        candidates_by_key[key] = apply_temporal_context(raw_candidates, config)
        counts_by_key[key] = estimate_count(config, video, train_videos, train_gt, [])
        audio = load_audio(args.data_root / video["video_path"], args.sample_rate)
        onsets_by_key[key] = audio_onset_by_frame(audio, args.sample_rate, int(video["frame_count"]))

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
    for window in parse_ints(args.audio_windows):
        for alpha in parse_floats(args.audio_alphas):
            rescored_by_key = {
                key: [
                    rescore_with_audio(candidate, onsets_by_key[key], alpha, window)
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
                    "window": window,
                    "alpha": alpha,
                    "changed": count_changed(baseline_rows, rows),
                }
            )

    print("score,delta,time,fighter,fp_penalty,wins,n_pred,changed,audio_window,alpha")
    for result in sorted(results, key=lambda item: item["macro"], reverse=True)[: args.top_k]:
        print(
            f"{result['macro']:.6f},{result['macro'] - baseline_score['macro_score']:.6f},"
            f"{result['time']:.6f},{result['fighter']:.6f},{result['fp_penalty']:.6f},"
            f"{result['wins']},{result['n_pred']},{result['changed']},"
            f"{result['window']},{result['alpha']}"
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


def rescore_with_audio(
    candidate: PunchCandidate,
    onset: np.ndarray,
    alpha: float,
    window: int,
) -> PunchCandidate:
    if window <= 0 or alpha == 0.0:
        return candidate
    lo = max(0, candidate.frame - window)
    hi = min(len(onset), candidate.frame + window + 1)
    local = float(np.max(onset[lo:hi])) if hi > lo else 0.0
    score = candidate.score * max(0.01, 1.0 + alpha * np.log1p(local))
    features = dict(candidate.features)
    features[f"audio_onset_w{window}"] = local
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
    print("root,rank,audio_window,alpha,macro_delta,time_delta,fp_delta,n_videos")
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
                f"{root},{rank},{result['window']},{result['alpha']},"
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


def parse_ints(values: str) -> list[int]:
    return [int(value) for value in values.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
