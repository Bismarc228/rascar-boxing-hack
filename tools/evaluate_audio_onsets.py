#!/usr/bin/env python3
"""Evaluate audio onset timing and audio-assisted pose timing shifts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import butter, filtfilt

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--pose-threshold", type=float, default=0.8)
    parser.add_argument("--pose-nms", type=int, default=6)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pool-thresholds", default="0.0")
    parser.add_argument("--pool-nms-frames", default="0")
    parser.add_argument("--fusion-thresholds", default="0.8,0.9,1.15")
    parser.add_argument("--fusion-nms-frames", default="4,6")
    parser.add_argument("--audio-windows", default="0,3,6")
    parser.add_argument("--audio-alphas", default="0.0,0.05,0.1,0.25")
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
    if args.max_videos:
        ready_keys = ready_keys[: args.max_videos]
    ready_set = set(ready_keys)
    print("ready=" + ",".join(ready_keys))

    gt_by_key: dict[str, list[dict[str, str]]] = {}
    for row in punches:
        if row["clear"] == "true" and row["video_key"] in ready_set:
            gt_by_key.setdefault(row["video_key"], []).append(row)
    train_gt = [row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set]
    attr_priors = fit_attr_priors(train_gt)

    onsets_by_key = {}
    peaks_by_key = {}
    candidates_by_key = {}
    pose_by_key = {}
    for key in progress(ready_keys, "audio+pose", args.quiet):
        video = video_by_key[key]
        audio = load_audio(args.data_root / video["video_path"], args.sample_rate)
        onset = audio_onset_by_frame(audio, args.sample_rate, int(video["frame_count"]))
        onsets_by_key[key] = onset
        peaks_by_key[key] = peak_frames(onset, min_distance=6)
        pose_candidates = score_pose_tracks(
            args.tracks_dir / f"{key}.jsonl",
            PoseHeuristicConfig(min_score=0.0),
        )
        candidates_by_key[key] = pose_candidates
        pose_by_key[key] = select_candidates(
            pose_candidates,
            PoseHeuristicConfig(min_score=args.pose_threshold, nms_frames=args.pose_nms),
            None,
        )

    baseline_rows = rows_from_candidates(ready_keys, video_by_key, pose_by_key, attr_priors)
    baseline_score = score_predictions([row for rows_ in gt_by_key.values() for row in rows_], baseline_rows)
    print_score_summary("baseline_pose", baseline_score, len(baseline_rows))

    print("audio_only_oracle_count")
    for min_distance in [4, 6, 8, 10, 12, 16, 20]:
        rows = []
        row_id = 1
        for key in ready_keys:
            video = video_by_key[key]
            count = len(gt_by_key.get(key, []))
            selected_frames = peak_frames(onsets_by_key[key], min_distance=min_distance)[:count]
            for frame in sorted(selected_frames):
                rows.append(default_row(row_id, video, frame))
                row_id += 1
        score = score_predictions([row for rows_ in gt_by_key.values() for row in rows_], rows)
        timing = timing_stats(gt_by_key, {key: peak_frames(onsets_by_key[key], min_distance)[: len(gt_by_key.get(key, []))] for key in ready_keys})
        print(
            f"dist={min_distance},n={len(rows)},score={score['macro_score']:.6f},"
            f"within15={timing['within15']:.3f},within30={timing['within30']:.3f},time={timing['time_score']:.3f}"
        )

    print("pose_shift_to_audio")
    for window in [0, 3, 6, 10, 15, 20]:
        rows = []
        row_id = 1
        shifted_count = 0
        for key in ready_keys:
            video = video_by_key[key]
            peaks = peaks_by_key[key]
            onset = onsets_by_key[key]
            for candidate in pose_by_key[key]:
                frame, shifted = shift_to_audio_peak(candidate.frame, peaks, onset, window)
                shifted_count += int(shifted)
                attrs = estimate_attrs(candidate, attr_priors)
                rows.append(
                    {
                        "id": str(row_id),
                        "video_id": video["video_id"],
                        "agn_index": video["agn_index"],
                        "video_key": key,
                        "frame": str(frame),
                        "fighter": candidate.fighter,
                        "punch_type": attrs["punch_type"],
                        "hand": candidate.hand,
                        "target": candidate.target,
                        "effectiveness": attrs["effectiveness"],
                        "clear": "true",
                    }
                )
                row_id += 1
        score = score_predictions([row for rows_ in gt_by_key.values() for row in rows_], rows)
        print(f"window={window},n={len(rows)},shifted={shifted_count},score={score['macro_score']:.6f}")

    print("pose_audio_rerank_filter")
    for alpha in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
        for keep_fraction in [0.55, 0.65, 0.75, 0.85, 1.0]:
            rows = []
            row_id = 1
            for key in ready_keys:
                video = video_by_key[key]
                rescored = [
                    rescore_with_audio(candidate, onsets_by_key[key], alpha)
                    for candidate in pose_by_key[key]
                ]
                count = round(len(rescored) * keep_fraction)
                selected = select_candidates(
                    rescored,
                    PoseHeuristicConfig(min_score=0.0, nms_frames=args.pose_nms),
                    count,
                )
                for candidate in selected:
                    attrs = estimate_attrs(candidate, attr_priors)
                    rows.append(
                        {
                            "id": str(row_id),
                            "video_id": video["video_id"],
                            "agn_index": video["agn_index"],
                            "video_key": key,
                            "frame": str(candidate.frame),
                            "fighter": candidate.fighter,
                            "punch_type": attrs["punch_type"],
                            "hand": candidate.hand,
                            "target": candidate.target,
                            "effectiveness": attrs["effectiveness"],
                            "clear": "true",
                        }
                    )
                    row_id += 1
            score = score_predictions([row for rows_ in gt_by_key.values() for row in rows_], rows)
            print(
                f"alpha={alpha},keep={keep_fraction},n={len(rows)},score={score['macro_score']:.6f}",
                flush=True,
            )

    print("pose_audio_wide_pool_grid")
    pool_thresholds = parse_floats(args.pool_thresholds)
    pool_nms_values = parse_ints(args.pool_nms_frames)
    fusion_thresholds = parse_floats(args.fusion_thresholds)
    fusion_nms_values = parse_ints(args.fusion_nms_frames)
    audio_windows = parse_ints(args.audio_windows)
    audio_alphas = parse_floats(args.audio_alphas)
    grid_results = []
    val_gt = [row for rows_ in gt_by_key.values() for row in rows_]
    for pool_threshold in pool_thresholds:
        for pool_nms in pool_nms_values:
            pooled_by_key = {
                key: select_candidates(
                    candidates_by_key[key],
                    PoseHeuristicConfig(min_score=pool_threshold, nms_frames=pool_nms),
                    None,
                )
                for key in ready_keys
            }
            for window in audio_windows:
                for alpha in audio_alphas:
                    rescored_by_key = {
                        key: [
                            rescore_with_audio(candidate, onsets_by_key[key], alpha, window)
                            for candidate in pooled_by_key[key]
                        ]
                        for key in ready_keys
                    }
                    for threshold in fusion_thresholds:
                        for nms_frames in fusion_nms_values:
                            selected_by_key = {
                                key: select_candidates(
                                    rescored_by_key[key],
                                    PoseHeuristicConfig(
                                        min_score=threshold,
                                        nms_frames=nms_frames,
                                    ),
                                    None,
                                )
                                for key in ready_keys
                            }
                            rows = rows_from_candidates(
                                ready_keys,
                                video_by_key,
                                selected_by_key,
                                attr_priors,
                            )
                            score = score_predictions(val_gt, rows)
                            summary = score_summary(score)
                            wins = video_wins(score, baseline_score)
                            grid_results.append(
                                (
                                    score["macro_score"],
                                    summary["time"],
                                    summary["fp_penalty"],
                                    wins,
                                    len(rows),
                                    pool_threshold,
                                    pool_nms,
                                    window,
                                    alpha,
                                    threshold,
                                    nms_frames,
                                )
                            )

    print(
        "score,time,fp_penalty,wins,n_pred,pool_threshold,pool_nms,"
        "audio_window,alpha,threshold,nms_frames"
    )
    for result in sorted(grid_results, reverse=True)[: args.top_k]:
        score, time_score, fp_penalty, wins, n_pred, pool_threshold, pool_nms, window, alpha, threshold, nms_frames = result
        print(
            f"{score:.6f},{time_score:.6f},{fp_penalty:.6f},{wins},{n_pred},"
            f"{pool_threshold},{pool_nms},{window},{alpha},{threshold},{nms_frames}"
        )

    return 0


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


def load_audio(video_path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    data = subprocess.check_output(cmd)
    return np.frombuffer(data, dtype=np.float32)


def audio_onset_by_frame(audio: np.ndarray, sample_rate: int, frame_count: int) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(frame_count, dtype=np.float32)
    # Punch impact is more transient in mid/high frequencies than in crowd noise.
    high = highpass(audio, sample_rate, cutoff=650.0)
    rms = frame_rms(audio, sample_rate, frame_count)
    high_rms = frame_rms(high, sample_rate, frame_count)
    log_rms = np.log1p(rms * 80.0)
    log_high = np.log1p(high_rms * 120.0)
    flux = positive_diff(log_high)
    energy_delta = positive_diff(log_rms)
    onset = 0.35 * zscore(log_high) + 0.45 * zscore(flux) + 0.20 * zscore(energy_delta)
    onset = np.maximum(0.0, onset)
    if len(onset) >= 3:
        onset = np.convolve(onset, np.array([0.25, 0.5, 0.25]), mode="same")
    return onset.astype(np.float32)


def highpass(audio: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
    if len(audio) < sample_rate:
        return audio
    b, a = butter(3, cutoff / (sample_rate * 0.5), btype="highpass")
    return filtfilt(b, a, audio).astype(np.float32)


def frame_rms(audio: np.ndarray, sample_rate: int, frame_count: int) -> np.ndarray:
    out = np.zeros(frame_count, dtype=np.float32)
    for frame in range(frame_count):
        start = round(frame * sample_rate / 30.0)
        end = round((frame + 1) * sample_rate / 30.0)
        if start >= len(audio):
            break
        chunk = audio[start : min(len(audio), max(end, start + 1))]
        if len(chunk):
            out[frame] = float(np.sqrt(np.mean(chunk * chunk)))
    return out


def positive_diff(values: np.ndarray) -> np.ndarray:
    diff = np.diff(values, prepend=values[:1])
    return np.maximum(0.0, diff)


def zscore(values: np.ndarray) -> np.ndarray:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return (values - med) / max(1e-6, 1.4826 * mad)


def peak_frames(values: np.ndarray, min_distance: int) -> list[int]:
    order = np.argsort(values)[::-1]
    selected = []
    for frame in order:
        frame_i = int(frame)
        if values[frame_i] <= 0:
            break
        if any(abs(frame_i - other) <= min_distance for other in selected):
            continue
        selected.append(frame_i)
    return selected


def shift_to_audio_peak(
    frame: int,
    peaks: list[int],
    onset: np.ndarray,
    window: int,
) -> tuple[int, bool]:
    if window <= 0:
        return frame, False
    nearby = [peak for peak in peaks if abs(peak - frame) <= window]
    if not nearby:
        return frame, False
    best = max(nearby, key=lambda item: onset[item])
    return int(best), int(best) != int(frame)


def rescore_with_audio(
    candidate: PunchCandidate,
    onset: np.ndarray,
    alpha: float,
    window: int = 3,
) -> PunchCandidate:
    lo = max(0, candidate.frame - window)
    hi = min(len(onset), candidate.frame + window + 1)
    if hi <= lo:
        local = 0.0
    else:
        local = float(np.max(onset[lo:hi]))
    score = candidate.score * max(0.01, 1.0 + alpha * np.log1p(local))
    features = dict(candidate.features)
    features["audio_onset_local"] = local
    features["audio_window"] = float(window)
    return PunchCandidate(
        candidate.video_key,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
        float(score),
        features,
    )


def default_row(row_id: int, video: dict[str, str], frame: int) -> dict[str, str]:
    return {
        "id": str(row_id),
        "video_id": video["video_id"],
        "agn_index": video["agn_index"],
        "video_key": video["video_key"],
        "frame": str(frame),
        "fighter": "blue",
        "punch_type": "hook",
        "hand": "left",
        "target": "head",
        "effectiveness": "landed",
        "clear": "true",
    }


def rows_from_candidates(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        for candidate in sorted(selected_by_key.get(key, []), key=lambda item: item.frame):
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


def print_score_summary(label: str, score: dict[str, object], n_pred: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},time={summary['time']:.6f},"
        f"fp_penalty={summary['fp_penalty']:.6f},n_pred={n_pred}"
    )


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def video_wins(score: dict[str, object], baseline: dict[str, object], eps: float = 1e-9) -> int:
    return sum(
        int(score["by_video"][key]["final_score"] > baseline["by_video"][key]["final_score"] + eps)
        for key in baseline["by_video"]
    )


def timing_stats(gt_by_key: dict[str, list[dict[str, str]]], pred_by_key: dict[str, list[int]]) -> dict[str, float]:
    total_gt = 0
    within15 = 0
    within30 = 0
    time_score = 0.0
    for key, gt_rows in gt_by_key.items():
        preds = pred_by_key.get(key, [])
        used = set()
        for row in gt_rows:
            total_gt += 1
            frame = int(row["frame"])
            best = None
            best_diff = 10**9
            for idx, pred in enumerate(preds):
                if idx in used:
                    continue
                diff = abs(pred - frame)
                if diff < best_diff:
                    best = idx
                    best_diff = diff
            if best is None or best_diff > 30:
                continue
            used.add(best)
            within30 += 1
            if best_diff <= 15:
                within15 += 1
                time_score += 1.0 - min(1.0, best_diff / 15.0)
    return {
        "within15": within15 / max(1, total_gt),
        "within30": within30 / max(1, total_gt),
        "time_score": time_score / max(1, total_gt),
    }


if __name__ == "__main__":
    raise SystemExit(main())
