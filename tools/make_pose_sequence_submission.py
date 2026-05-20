#!/usr/bin/env python3
"""Train a pose-sequence spotter on cached labeled tracks and write test CSV."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from rascar_boxing.validation import validate_submission
from tools.evaluate_pose_selection_variants import estimate_count
from tools.evaluate_pose_sequence_spotter import (
    GROUP_INDEX,
    TinyTCN,
    VideoPack,
    apply_pose_prior,
    build_stream_features,
    build_stream_labels,
    complete_track_keys,
    fit_standardizer,
    make_samples,
    parse_ints,
    predict_samples,
    replace_score,
    resolve_device,
    seed_everything,
    snap_frame,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-tracks-dir", type=Path, required=True)
    parser.add_argument("--train-witness-tracks-dirs", default="")
    parser.add_argument("--test-tracks-dir", type=Path, required=True)
    parser.add_argument("--test-witness-tracks-dirs", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=2200)
    parser.add_argument("--label-sigma", type=float, default=5.0)
    parser.add_argument("--near-window", type=int, default=30)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--chunks-per-epoch", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--pos-sample-rate", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--pose-prior", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-frames", type=int, default=10)
    parser.add_argument("--cross-nms-frames", type=int, default=2)
    parser.add_argument("--snap-window", type=int, default=0)
    parser.add_argument("--count-mode", default="root_count")
    parser.add_argument("--count-multiplier", type=float, default=0.92)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    seed_everything(args.seed, args.deterministic)

    train_videos = read_csv_rows(args.data_root / "train/videos.csv")
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    punches = [row for row in read_csv_rows(args.data_root / "train/punches.csv") if row["clear"] == "true"]
    sample_rows = read_csv_rows(args.data_root / "sample_submission.csv")
    train_video_by_key = {row["video_key"]: row for row in train_videos}
    test_video_by_key = {row["video_key"]: row for row in test_videos}
    train_keys = complete_track_keys(args.train_tracks_dir, train_video_by_key)
    test_keys = complete_track_keys(args.test_tracks_dir, test_video_by_key)
    if not train_keys:
        print("No complete labeled train tracks found")
        return 2
    if set(test_keys) != {row["video_key"] for row in test_videos}:
        print(f"Missing complete test tracks: {sorted({row['video_key'] for row in test_videos} - set(test_keys))}")
        return 2

    train_witness_dirs = [Path(item) for item in args.train_witness_tracks_dirs.split(",") if item]
    test_witness_dirs = [Path(item) for item in args.test_witness_tracks_dirs.split(",") if item]
    if len(train_witness_dirs) != len(test_witness_dirs):
        raise ValueError("Train/test witness dir counts must match")

    gt_by_key: dict[str, list[dict[str, str]]] = {}
    for row in punches:
        gt_by_key.setdefault(row["video_key"], []).append(row)

    train_packs = {
        key: build_labeled_pack(
            key,
            train_video_by_key[key],
            args.train_tracks_dir,
            train_witness_dirs,
            gt_by_key.get(key, []),
            args,
        )
        for key in train_keys
    }
    test_packs = {
        key: build_unlabeled_pack(
            key,
            test_video_by_key[key],
            args.test_tracks_dir,
            test_witness_dirs,
            args,
        )
        for key in test_keys
    }
    train_samples = make_samples(train_packs, train_keys)
    test_samples = make_samples(test_packs, test_keys)
    mean, std = fit_standardizer(train_samples)
    for sample in train_samples + test_samples:
        sample.x = ((sample.x - mean) / std).astype(np.float32)

    device = resolve_device(args.device)
    seed_values = parse_ints(args.seeds) if args.seeds else [args.seed]
    print(f"train_keys={len(train_keys)} {','.join(train_keys)}")
    print(f"test_keys={len(test_keys)} {','.join(test_keys)}")
    print(f"device={device} seeds={','.join(str(seed) for seed in seed_values)}")

    prediction_sum: list[np.ndarray] | None = None
    for seed_index, model_seed in enumerate(seed_values):
        seed_everything(model_seed, args.deterministic)
        model = TinyTCN(train_samples[0].x.shape[1], args.hidden, args.layers, args.dropout).to(device)
        train_model(
            model,
            train_samples,
            args,
            device,
            fold_index=0,
            group="all_train",
            sample_seed=model_seed,
            seed_index=seed_index,
        )
        predictions = predict_samples(model, test_samples, device)
        if prediction_sum is None:
            prediction_sum = [prediction.copy() for prediction in predictions]
        else:
            for index, prediction in enumerate(predictions):
                prediction_sum[index] += prediction
    assert prediction_sum is not None
    predictions = [prediction / len(seed_values) for prediction in prediction_sum]
    by_key_group = {
        (sample.key, sample.group_index): prediction
        for sample, prediction in zip(test_samples, predictions)
    }

    train_counts = Counter(row["video_key"] for row in punches)
    attr_priors = fit_attr_priors(punches)
    selected_by_video: dict[str, list[PunchCandidate]] = {}
    for video in sorted(test_videos, key=lambda row: row["video_key"]):
        key = video["video_key"]
        scored = [
            replace_score(
                candidate,
                float(
                    by_key_group[(key, GROUP_INDEX[(candidate.fighter, candidate.hand)])][
                        min(
                            candidate.frame,
                            len(by_key_group[(key, GROUP_INDEX[(candidate.fighter, candidate.hand)])]) - 1,
                        )
                    ]
                ),
            )
            for candidate in test_packs[key].candidates
        ]
        scored = [apply_pose_prior(candidate, args.pose_prior) for candidate in scored]
        count = estimate_count(
            video,
            train_videos,
            train_counts,
            Counter(),
            args.count_mode,
            args.count_multiplier,
        )
        selected = select_candidates(
            scored,
            PoseHeuristicConfig(
                min_score=args.threshold,
                nms_frames=args.nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=args.cross_nms_frames,
            ),
            count,
        )
        selected_by_video[key] = snap_selected_candidates(
            selected,
            video,
            args.snap_window,
            by_key_group,
        )

    rows = fill_sample_rows(sample_rows, test_videos, selected_by_video, attr_priors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output, rows, SUBMISSION_COLUMNS)
    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1

    by_video = Counter(row["video_key"] for row in rows if row["clear"] == "true")
    print(f"Wrote {args.output}")
    print(
        "config="
        f"threshold={args.threshold},nms={args.nms_frames},cross={args.cross_nms_frames},"
        f"count_mode={args.count_mode},count_multiplier={args.count_multiplier},"
        f"pose_prior={args.pose_prior},snap_window={args.snap_window}"
    )
    print("selected=" + ",".join(f"{key}:{by_video.get(key, 0)}" for key in sorted(test_video_by_key)))
    print(f"total_clear={sum(by_video.values())}")
    return 0


def build_labeled_pack(
    key: str,
    video: dict[str, str],
    primary_dir: Path,
    witness_dirs: list[Path],
    gt_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> VideoPack:
    candidates, streams = build_candidates_and_streams(key, video, primary_dir, witness_dirs, args)
    labels, weights = build_stream_labels(video, gt_rows, args.label_sigma, args.near_window)
    return VideoPack(key, video, candidates, streams, labels, weights)


def build_unlabeled_pack(
    key: str,
    video: dict[str, str],
    primary_dir: Path,
    witness_dirs: list[Path],
    args: argparse.Namespace,
) -> VideoPack:
    candidates, streams = build_candidates_and_streams(key, video, primary_dir, witness_dirs, args)
    frame_count = int(video["frame_count"])
    labels = [np.zeros(frame_count, dtype=np.float32) for _ in range(4)]
    weights = [np.ones(frame_count, dtype=np.float32) for _ in range(4)]
    return VideoPack(key, video, candidates, streams, labels, weights)


def build_candidates_and_streams(
    key: str,
    video: dict[str, str],
    primary_dir: Path,
    witness_dirs: list[Path],
    args: argparse.Namespace,
) -> tuple[list[PunchCandidate], list[np.ndarray]]:
    primary_raw = score_pose_tracks(primary_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
    primary_pool = select_candidates(
        primary_raw,
        PoseHeuristicConfig(
            min_score=args.pool_min_score,
            nms_frames=args.pool_nms_frames,
            nms_group_mode="fighter_hand",
            cross_nms_frames=0,
        ),
        args.max_candidates_per_video,
    )
    model_candidate_sets = [primary_raw]
    for witness_dir in witness_dirs:
        model_candidate_sets.append(
            score_pose_tracks(witness_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        )
    streams = build_stream_features(video, model_candidate_sets)
    return primary_pool, streams


def snap_selected_candidates(
    selected: list[PunchCandidate],
    video: dict[str, str],
    snap_window: int,
    by_key_group: dict[tuple[str, int], np.ndarray],
) -> list[PunchCandidate]:
    if snap_window <= 0:
        return selected
    snapped = []
    for candidate in selected:
        frame = snap_frame(candidate, video, snap_window, by_key_group)
        snapped.append(
            PunchCandidate(
                candidate.video_key,
                frame,
                candidate.fighter,
                candidate.hand,
                candidate.target,
                candidate.score,
                candidate.features,
            )
        )
    return sorted(snapped, key=lambda item: item.frame)


def fill_sample_rows(
    sample_rows: list[dict[str, str]],
    videos: list[dict[str, str]],
    selected_by_video: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
) -> list[dict[str, str]]:
    videos_by_key = {row["video_key"]: row for row in videos}
    used_by_video: Counter[str] = Counter()
    output = []
    for sample_row in sample_rows:
        row = {column: sample_row[column] for column in SUBMISSION_COLUMNS}
        video_key = row["video_key"]
        index = used_by_video[video_key]
        selected = selected_by_video.get(video_key, [])
        if index < len(selected):
            candidate = selected[index]
            video = videos_by_key[video_key]
            attrs = estimate_attrs(candidate, attr_priors)
            row["frame"] = str(max(0, min(int(video["frame_count"]) - 1, candidate.frame)))
            row["fighter"] = candidate.fighter
            row["punch_type"] = attrs["punch_type"]
            row["hand"] = candidate.hand
            row["target"] = candidate.target
            row["effectiveness"] = attrs["effectiveness"]
            row["clear"] = "true"
        else:
            row["clear"] = "false"
        used_by_video[video_key] += 1
        output.append(row)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
