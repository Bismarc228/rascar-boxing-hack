#!/usr/bin/env python3
"""Evaluate audio onset features as a fixed-row keep/drop gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_audio_onsets import (
    audio_onset_by_frame,
    frame_rms,
    highpass,
    load_audio,
    peak_frames,
    positive_diff,
    zscore,
)
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
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--feature-mode", choices=["audio", "pose_audio", "pose"], default="pose_audio")
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--pose-feature-windows", default="4,8,16")
    parser.add_argument("--audio-feature-windows", default="0,3,6,12,24")
    parser.add_argument("--audio-offsets", default="-12,-6,-3,0,3,6,12")
    parser.add_argument(
        "--thresholds",
        default="0.1,0.15,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.35,0.4,0.5,0.6",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
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
    indexed_by_key = {}
    if args.feature_mode in {"pose", "pose_audio"}:
        indexed_by_key = {
            key: index_candidates(
                apply_temporal_context(
                    score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config),
                    config,
                )
            )
            for key in keys
        }
    audio_by_key = {}
    if args.feature_mode in {"audio", "pose_audio"}:
        audio_by_key = {
            key: compute_audio_tracks(
                args.data_root / video_by_key[key]["video_path"],
                args.sample_rate,
                as_int(video_by_key[key]["frame_count"], "frame_count"),
            )
            for key in progress(keys, "audio-features", args.quiet)
        }

    pose_windows = parse_ints(args.pose_feature_windows)
    audio_windows = parse_ints(args.audio_feature_windows)
    audio_offsets = parse_ints(args.audio_offsets)
    x = np.stack(
        [
            combined_features(
                row,
                video_by_key[row["video_key"]],
                indexed_by_key.get(row["video_key"], {}),
                audio_by_key.get(row["video_key"]),
                args.match_window,
                pose_windows,
                audio_windows,
                audio_offsets,
                args.feature_mode,
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

    print("score,delta,fighter,time,fp,wins,n_kept,n_dropped,threshold,feature_mode,model")
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print(
            f"{item['score']:.6f},{item['delta']:.6f},{item['fighter']:.6f},"
            f"{item['time']:.6f},{item['fp']:.6f},{item['wins']},"
            f"{item['n_kept']},{item['n_dropped']},{item['threshold']},"
            f"{args.feature_mode},{args.model}"
        )
    if args.write_oof_rows:
        threshold = args.write_threshold
        if threshold is None:
            threshold = max(results, key=lambda row: row["score"])["threshold"]
        rows = apply_keep_mask(pred_rows, p_keep >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} threshold={threshold} n_rows={len(rows)}")
    return 0


def compute_audio_tracks(video_path: Path, sample_rate: int, frame_count: int) -> dict[str, np.ndarray | list[int]]:
    audio = load_audio(video_path, sample_rate)
    high = highpass(audio, sample_rate, cutoff=650.0) if len(audio) else audio
    rms = frame_rms(audio, sample_rate, frame_count) if len(audio) else np.zeros(frame_count, dtype=np.float32)
    high_rms = frame_rms(high, sample_rate, frame_count) if len(audio) else np.zeros(frame_count, dtype=np.float32)
    log_rms = np.log1p(rms * 80.0)
    log_high = np.log1p(high_rms * 120.0)
    flux = positive_diff(log_high)
    energy_delta = positive_diff(log_rms)
    onset = audio_onset_by_frame(audio, sample_rate, frame_count)
    peaks = peak_frames(onset, min_distance=6)
    peak_rank = np.full(frame_count, 1.0, dtype=np.float32)
    for rank, frame in enumerate(peaks[: max(1, frame_count)], start=1):
        if 0 <= frame < frame_count:
            peak_rank[frame] = rank / max(1.0, float(len(peaks)))
    return {
        "onset": onset,
        "rms": rms.astype(np.float32),
        "high_rms": high_rms.astype(np.float32),
        "flux": flux.astype(np.float32),
        "energy_delta": energy_delta.astype(np.float32),
        "peaks": peaks,
        "peak_rank": peak_rank,
    }


def combined_features(
    row: dict[str, str],
    video: dict[str, str],
    candidates_by_frame,
    audio_tracks: dict[str, np.ndarray | list[int]] | None,
    match_window: int,
    pose_windows: list[int],
    audio_windows: list[int],
    audio_offsets: list[int],
    feature_mode: str,
) -> np.ndarray:
    values: list[float] = []
    if feature_mode in {"pose", "pose_audio"}:
        values.extend(row_features(row, video, candidates_by_frame, match_window, pose_windows).tolist())
    if feature_mode in {"audio", "pose_audio"}:
        if audio_tracks is None:
            raise ValueError("audio_tracks is required for audio feature modes")
        values.extend(audio_features(row, video, audio_tracks, audio_windows, audio_offsets))
    return np.asarray(values, dtype=np.float32)


def audio_features(
    row: dict[str, str],
    video: dict[str, str],
    tracks: dict[str, np.ndarray | list[int]],
    windows: list[int],
    offsets: list[int],
) -> list[float]:
    frame = as_int(row["frame"], "frame")
    frame_count = as_int(video["frame_count"], "frame_count")
    values = [
        frame / max(1.0, float(frame_count)),
        1.0 if row["fighter"] == "red" else 0.0,
        1.0 if row["hand"] == "left" else 0.0,
        1.0 if row["target"] == "head" else 0.0,
    ]
    for root in ["бокс", "Турнир Бокс", "Турнир Бокс 2"]:
        values.append(float(video["data_root"] == root))

    for name in ["onset", "rms", "high_rms", "flux", "energy_delta"]:
        arr = tracks[name]
        assert isinstance(arr, np.ndarray)
        for offset in offsets:
            idx = max(0, min(frame_count - 1, frame + offset))
            raw = float(arr[idx]) if idx < len(arr) else 0.0
            values.append(raw)
            values.append(float(np.log1p(max(0.0, raw))))

    onset = tracks["onset"]
    peak_rank = tracks["peak_rank"]
    peaks = tracks["peaks"]
    assert isinstance(onset, np.ndarray)
    assert isinstance(peak_rank, np.ndarray)
    assert isinstance(peaks, list)
    positive = onset[onset > 0]
    for window in windows:
        lo = max(0, frame - window)
        hi = min(len(onset), frame + window + 1)
        local = onset[lo:hi] if hi > lo else np.asarray([], dtype=np.float32)
        if len(local):
            local_max = float(local.max())
            local_mean = float(local.mean())
            local_std = float(local.std())
            center = float(onset[min(frame, len(onset) - 1)])
        else:
            local_max = local_mean = local_std = center = 0.0
        nearby = [peak for peak in peaks if abs(peak - frame) <= window]
        if nearby:
            best_peak = max(nearby, key=lambda item: float(onset[item]))
            nearest_peak = min(nearby, key=lambda item: abs(item - frame))
            best_delta = float(best_peak - frame)
            nearest_delta = float(nearest_peak - frame)
            best_value = float(onset[best_peak])
            nearest_rank = float(peak_rank[nearest_peak])
        else:
            best_delta = nearest_delta = 999.0
            best_value = 0.0
            nearest_rank = 1.0
        percentile = float((positive <= local_max).mean()) if len(positive) else 0.0
        values.extend(
            [
                local_max,
                local_mean,
                local_std,
                center,
                local_max - center,
                percentile,
                min(abs(best_delta), 60.0) / 60.0,
                np.sign(best_delta) * min(abs(best_delta), 60.0) / 60.0,
                min(abs(nearest_delta), 60.0) / 60.0,
                np.sign(nearest_delta) * min(abs(nearest_delta), 60.0) / 60.0,
                best_value,
                nearest_rank,
            ]
        )
    return values


def progress(items: list[str], desc: str, quiet: bool):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc, leave=False, disable=quiet, mininterval=5.0)
    except Exception:
        if not quiet:
            print(f"{desc}: {len(items)} items")
        return items


if __name__ == "__main__":
    raise SystemExit(main())
