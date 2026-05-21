#!/usr/bin/env python3
"""Evaluate a small pose-sequence spotter on cached pose tracks.

The model predicts punch probability per frame for each (fighter, hand) stream.
Only primary pose candidates are emitted after OOF inference, so this is a
sequence scorer/reranker rather than a standalone dense detector.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Needed for deterministic CuBLAS kernels when deterministic algorithms are on.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from tools.evaluate_pose_selection_variants import estimate_count, score_summary, video_wins
from tools.evaluate_rgb_contact_clip import (
    fit_oof_temporal_heads as fit_rgb_contact_heads,
    load_or_extract_features as load_or_extract_rgb_contact_features,
)


GROUPS = [("red", "left"), ("red", "right"), ("blue", "left"), ("blue", "right")]
GROUP_INDEX = {group: index for index, group in enumerate(GROUPS)}
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


@dataclass
class VideoPack:
    key: str
    video: dict[str, str]
    candidates: list[PunchCandidate]
    streams: list[np.ndarray]
    labels: list[np.ndarray]
    weights: list[np.ndarray]


@dataclass
class StreamSample:
    key: str
    group_index: int
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    positive_frames: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--witness-tracks-dirs", default="")
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=1800)
    parser.add_argument("--label-sigma", type=float, default=5.0)
    parser.add_argument("--near-window", type=int, default=30)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--chunks-per-epoch", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--pos-sample-rate", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--thresholds", default="0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4")
    parser.add_argument("--nms-frames", default="8,10,12")
    parser.add_argument("--cross-nms-frames", default="2,4")
    parser.add_argument("--snap-windows", default="0")
    parser.add_argument("--count-modes", default="threshold,root_rate,root_count")
    parser.add_argument("--count-multipliers", default="0.72,0.78,0.84,0.88,0.92")
    parser.add_argument("--pose-priors", default="0.0,0.1,0.2,0.4")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-best-rows", type=Path)
    parser.add_argument("--rgb-contact-feature-cache", type=Path)
    parser.add_argument("--rgb-contact-model-name", default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--rgb-contact-image-size", type=int, default=224)
    parser.add_argument("--rgb-contact-batch-size", type=int, default=96)
    parser.add_argument("--rgb-contact-clip-len", type=int, default=4)
    parser.add_argument("--rgb-contact-frame-stride", type=int, default=2)
    parser.add_argument("--rgb-contact-crop-modes", default="union")
    parser.add_argument("--rgb-contact-crop-expand", type=float, default=0.16)
    parser.add_argument("--rgb-contact-decode-mode", choices=["sequential", "seek"], default="sequential")
    parser.add_argument("--rgb-contact-epochs", type=int, default=12)
    parser.add_argument("--rgb-contact-head-batch-size", type=int, default=256)
    parser.add_argument("--rgb-contact-hidden", type=int, default=128)
    parser.add_argument("--rgb-contact-dropout", type=float, default=0.15)
    parser.add_argument("--rgb-contact-lr", type=float, default=2e-3)
    parser.add_argument("--rgb-contact-weight-decay", type=float, default=2e-3)
    parser.add_argument("--rgb-contact-offset-loss-weight", type=float, default=0.10)
    parser.add_argument("--rgb-contact-offset-scale", type=float, default=12.0)
    parser.add_argument("--rgb-contact-label-window", type=int, default=12)
    parser.add_argument("--rgb-contact-blend-alphas", default="0.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    seed_everything(args.seed, args.deterministic)

    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if len(ready_keys) < 3:
        print(f"Need at least 3 complete tracks, got {ready_keys}")
        return 2

    witness_dirs = [Path(item) for item in args.witness_tracks_dirs.split(",") if item]
    witness_dirs = [
        path
        for path in witness_dirs
        if set(ready_keys).issubset(set(complete_track_keys(path, video_by_key)))
    ]
    if witness_dirs and not args.quiet:
        print("witness_dirs=" + ",".join(str(path) for path in witness_dirs), flush=True)

    clear_gt = [row for row in punches if row["clear"] == "true"]
    gt_by_key: dict[str, list[dict[str, str]]] = {}
    for row in clear_gt:
        gt_by_key.setdefault(row["video_key"], []).append(row)

    packs: dict[str, VideoPack] = {}
    for key in ready_keys:
        primary_raw = score_pose_tracks(
            args.tracks_dir / f"{key}.jsonl",
            PoseHeuristicConfig(min_score=0.0),
        )
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
        streams = build_stream_features(video_by_key[key], model_candidate_sets)
        labels, weights = build_stream_labels(
            video_by_key[key],
            gt_by_key.get(key, []),
            args.label_sigma,
            args.near_window,
        )
        packs[key] = VideoPack(key, video_by_key[key], primary_pool, streams, labels, weights)

    print(f"ready={len(ready_keys)} {','.join(ready_keys)}", flush=True)
    print("pool=" + ",".join(f"{key}:{len(packs[key].candidates)}" for key in ready_keys), flush=True)

    group_by_key = {key: fight_group(video_by_key[key]) for key in ready_keys}
    groups = sorted(set(group_by_key.values()))
    scored_by_key: dict[str, list[PunchCandidate]] = {}
    stream_scores_by_key_group: dict[tuple[str, int], np.ndarray] = {}
    device = resolve_device(args.device)
    seed_values = parse_ints(args.seeds) if args.seeds else [args.seed]
    print(f"device={device}", flush=True)
    print("seeds=" + ",".join(str(seed) for seed in seed_values), flush=True)

    for fold_index, group in enumerate(groups):
        valid_keys = [key for key in ready_keys if group_by_key[key] == group]
        train_keys = [key for key in ready_keys if group_by_key[key] != group]
        train_samples = make_samples(packs, train_keys)
        valid_samples = make_samples(packs, valid_keys)
        mean, std = fit_standardizer(train_samples)
        for sample in train_samples + valid_samples:
            sample.x = ((sample.x - mean) / std).astype(np.float32)

        stream_predictions_sum: list[np.ndarray] | None = None
        for seed_index, model_seed in enumerate(seed_values):
            seed_everything(model_seed + 1000 * fold_index, args.deterministic)
            model = TinyTCN(train_samples[0].x.shape[1], args.hidden, args.layers, args.dropout).to(device)
            train_model(
                model,
                train_samples,
                args,
                device,
                fold_index,
                group,
                sample_seed=model_seed + 1000 * fold_index,
                seed_index=seed_index,
            )
            stream_predictions = predict_samples(model, valid_samples, device)
            if stream_predictions_sum is None:
                stream_predictions_sum = [prediction.copy() for prediction in stream_predictions]
            else:
                for index, prediction in enumerate(stream_predictions):
                    stream_predictions_sum[index] += prediction
        assert stream_predictions_sum is not None
        stream_predictions = [prediction / len(seed_values) for prediction in stream_predictions_sum]
        by_key_group = {
            (sample.key, sample.group_index): prediction
            for sample, prediction in zip(valid_samples, stream_predictions)
        }
        stream_scores_by_key_group.update(by_key_group)
        for key in valid_keys:
            scored_by_key[key] = [
                replace_score(
                    candidate,
                    float(
                        by_key_group[(key, GROUP_INDEX[(candidate.fighter, candidate.hand)])][
                            min(candidate.frame, len(by_key_group[(key, GROUP_INDEX[(candidate.fighter, candidate.hand)])]) - 1)
                        ]
                    ),
                )
                for candidate in packs[key].candidates
            ]
        print(
            f"fold={fold_index + 1}/{len(groups)} group={group} "
            f"train={len(train_keys)} valid={','.join(valid_keys)}",
            flush=True,
        )

    if set(scored_by_key) != set(ready_keys):
        print(f"Missing OOF predictions for {sorted(set(ready_keys) - set(scored_by_key))}")
        return 2
    rgb_contact_by_key = None
    if args.rgb_contact_feature_cache:
        rgb_contact_by_key = compute_rgb_contact_probabilities(
            args,
            ready_keys,
            video_by_key,
            packs,
            gt_by_key,
        )

    train_gt = [row for row in clear_gt if row["video_key"] not in set(ready_keys)]
    gt = [row for row in clear_gt if row["video_key"] in set(ready_keys)]
    train_videos = [row for row in videos if row["video_key"] not in set(ready_keys)]
    train_counts = Counter(row["video_key"] for row in train_gt)
    gt_counts = Counter(row["video_key"] for row in gt)
    attr_priors = fit_attr_priors(train_gt)

    baseline_rows = build_rows(
        ready_keys,
        video_by_key,
        {key: packs[key].candidates for key in ready_keys},
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        threshold=0.65,
        nms_frames=8,
        cross_nms=4,
        count_mode="threshold",
        count_multiplier=1.0,
        pose_prior=0.0,
        snap_window=0,
        stream_scores_by_key_group=stream_scores_by_key_group,
    )
    baseline_score = score_predictions(gt, baseline_rows)
    print_score("primary_pool_ref", baseline_score, len(baseline_rows), 0)

    results = []
    thresholds = parse_floats(args.thresholds)
    nms_values = parse_ints(args.nms_frames)
    cross_values = parse_ints(args.cross_nms_frames)
    snap_windows = parse_ints(args.snap_windows)
    count_modes = [value for value in args.count_modes.split(",") if value]
    multipliers = parse_floats(args.count_multipliers)
    pose_priors = parse_floats(args.pose_priors)

    for pose_prior in pose_priors:
        pose_reranked = {
            key: [apply_pose_prior(candidate, pose_prior) for candidate in candidates]
            for key, candidates in scored_by_key.items()
        }
        rgb_alphas = [0.0] if rgb_contact_by_key is None else parse_floats(args.rgb_contact_blend_alphas)
        for rgb_alpha in rgb_alphas:
            reranked = apply_rgb_contact_prior(pose_reranked, rgb_contact_by_key, rgb_alpha)
            for threshold in thresholds:
                for nms_frames in nms_values:
                    for cross_nms in cross_values:
                        for snap_window in snap_windows:
                            for count_mode in count_modes:
                                mode_multipliers = [1.0] if count_mode == "threshold" else multipliers
                                for count_multiplier in mode_multipliers:
                                    rows = build_rows(
                                        ready_keys,
                                        video_by_key,
                                        reranked,
                                        attr_priors,
                                        train_videos,
                                        train_counts,
                                        gt_counts,
                                        threshold,
                                        nms_frames,
                                        cross_nms,
                                        count_mode,
                                        count_multiplier,
                                        pose_prior=0.0,
                                        snap_window=snap_window,
                                        stream_scores_by_key_group=stream_scores_by_key_group,
                                    )
                                    score = score_predictions(gt, rows)
                                    summary = score_summary(score)
                                    results.append(
                                        (
                                            score["macro_score"],
                                            summary["time"],
                                            summary["fp_penalty"],
                                            video_wins(score, baseline_score),
                                            len(rows),
                                            pose_prior,
                                            rgb_alpha,
                                            threshold,
                                            nms_frames,
                                            cross_nms,
                                            snap_window,
                                            count_mode,
                                            count_multiplier,
                                        )
                                    )

    print(
        "score,time,fp_penalty,wins,n_pred,pose_prior,rgb_alpha,threshold,nms,cross_nms,snap_window,"
        "count_mode,count_multiplier"
    )
    for result in sorted(results, reverse=True)[: args.top_k]:
        print(
            f"{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},{result[3]},"
            f"{result[4]},{result[5]},{result[6]},{result[7]},{result[8]},"
            f"{result[9]},{result[10]},{result[11]},{result[12]}"
        )
    best_result = sorted(results, reverse=True)[0]
    best_rows = build_result_rows(
        best_result,
        ready_keys,
        video_by_key,
        scored_by_key,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        stream_scores_by_key_group,
        rgb_contact_by_key,
    )
    print_best_detail(
        best_result,
        ready_keys,
        video_by_key,
        gt,
        best_rows,
    )
    if args.write_best_rows:
        write_csv_rows(args.write_best_rows, best_rows, SUBMISSION_COLUMNS)
        print(f"wrote_best_rows={args.write_best_rows}", flush=True)
    return 0


def complete_track_keys(tracks_dir: Path, video_by_key: dict[str, dict[str, str]]) -> list[str]:
    keys = []
    for path in sorted(tracks_dir.glob("*.jsonl")):
        key = path.stem
        if key not in video_by_key:
            continue
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count == int(video_by_key[key]["frame_count"]):
            keys.append(key)
    return keys


def build_stream_features(
    video: dict[str, str],
    model_candidate_sets: list[list[PunchCandidate]],
) -> list[np.ndarray]:
    frame_count = int(video["frame_count"])
    frame_pos = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    all_model_arrays = []
    for candidates in model_candidate_sets:
        all_model_arrays.append(model_frame_arrays(candidates, frame_count))

    streams = []
    for group_index, (fighter, hand) in enumerate(GROUPS):
        cols = [
            frame_pos,
            np.full(frame_count, 1.0 if fighter == "red" else 0.0, dtype=np.float32),
            np.full(frame_count, 1.0 if hand == "left" else 0.0, dtype=np.float32),
        ]
        for model_arrays in all_model_arrays:
            group_score = model_arrays["group_score"][group_index]
            group_count = model_arrays["group_count"][group_index]
            all_score = model_arrays["all_score"]
            same_fighter_other_hand = model_arrays["group_score"][other_hand_index(group_index)]
            other_fighter_same_hand = model_arrays["group_score"][other_fighter_index(group_index)]
            cols.extend(
                [
                    np.log1p(group_score),
                    group_count,
                    np.log1p(all_score),
                    np.log1p(same_fighter_other_hand),
                    np.log1p(other_fighter_same_hand),
                ]
            )
            for window in (2, 4, 8, 16):
                cols.append(rolling_sum(np.log1p(group_score), window))
                cols.append(rolling_sum(np.log1p(all_score), window))
            if model_arrays["features"] is not None:
                for feature_array in model_arrays["features"][group_index]:
                    cols.append(feature_array)
        streams.append(np.stack(cols, axis=1).astype(np.float32))
    return streams


def model_frame_arrays(candidates: list[PunchCandidate], frame_count: int) -> dict[str, np.ndarray]:
    group_score = np.zeros((len(GROUPS), frame_count), dtype=np.float32)
    group_count = np.zeros((len(GROUPS), frame_count), dtype=np.float32)
    feature_values = np.zeros((len(GROUPS), len(BASE_FEATURES), frame_count), dtype=np.float32)
    feature_score = np.zeros((len(GROUPS), frame_count), dtype=np.float32)

    for candidate in candidates:
        group = GROUP_INDEX[(candidate.fighter, candidate.hand)]
        frame = max(0, min(frame_count - 1, candidate.frame))
        score = max(0.0, float(candidate.score))
        group_score[group, frame] += score
        group_count[group, frame] += 1.0
        if score >= feature_score[group, frame]:
            feature_score[group, frame] = score
            for index, name in enumerate(BASE_FEATURES):
                value = float(candidate.features.get(name, 0.0))
                if name == "forward_norm":
                    value /= 1000.0
                feature_values[group, index, frame] = value

    return {
        "group_score": group_score,
        "group_count": group_count,
        "all_score": group_score.sum(axis=0),
        "features": feature_values,
    }


def build_stream_labels(
    video: dict[str, str],
    gt_rows: list[dict[str, str]],
    sigma: float,
    near_window: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frame_count = int(video["frame_count"])
    labels = [np.zeros(frame_count, dtype=np.float32) for _ in GROUPS]
    near_any = np.zeros(frame_count, dtype=np.float32)
    frames = np.arange(frame_count, dtype=np.float32)
    for row in gt_rows:
        frame = int(row["frame"])
        group_index = GROUP_INDEX[(row["fighter"], row["hand"])]
        lo = max(0, frame - int(near_window))
        hi = min(frame_count, frame + int(near_window) + 1)
        near_any[lo:hi] = 1.0
        lo = max(0, int(frame - sigma * 4))
        hi = min(frame_count, int(frame + sigma * 4) + 1)
        values = np.exp(-0.5 * ((frames[lo:hi] - frame) / max(1e-3, sigma)) ** 2)
        labels[group_index][lo:hi] = np.maximum(labels[group_index][lo:hi], values)

    weights = []
    for label in labels:
        weights.append((1.0 + 24.0 * label + 2.0 * np.maximum(0.0, near_any - label)).astype(np.float32))
    return labels, weights


def make_samples(packs: dict[str, VideoPack], keys: list[str]) -> list[StreamSample]:
    samples = []
    for key in keys:
        pack = packs[key]
        for group_index, x in enumerate(pack.streams):
            positive_frames = np.flatnonzero(pack.labels[group_index] > 0.2).astype(np.int32)
            samples.append(
                StreamSample(
                    key=key,
                    group_index=group_index,
                    x=x,
                    y=pack.labels[group_index],
                    weight=pack.weights[group_index],
                    positive_frames=positive_frames,
                )
            )
    return samples


class ChunkDataset(Dataset):
    def __init__(
        self,
        samples: list[StreamSample],
        chunk_len: int,
        chunks_per_epoch: int,
        pos_sample_rate: float,
        seed: int,
    ) -> None:
        self.samples = samples
        self.chunk_len = chunk_len
        self.chunks_per_epoch = chunks_per_epoch
        self.pos_sample_rate = pos_sample_rate
        self.rng = random.Random(seed)
        self.positive_samples = [sample for sample in samples if len(sample.positive_frames)]

    def __len__(self) -> int:
        return self.chunks_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_positive = self.positive_samples and self.rng.random() < self.pos_sample_rate
        sample = self.rng.choice(self.positive_samples if use_positive else self.samples)
        length = len(sample.y)
        if length <= self.chunk_len:
            start = 0
        elif use_positive:
            center = int(self.rng.choice(sample.positive_frames))
            start = max(0, min(length - self.chunk_len, center - self.rng.randint(0, self.chunk_len - 1)))
        else:
            start = self.rng.randint(0, length - self.chunk_len)
        end = min(length, start + self.chunk_len)
        x = sample.x[start:end]
        y = sample.y[start:end]
        weight = sample.weight[start:end]
        if len(y) < self.chunk_len:
            pad = self.chunk_len - len(y)
            x = np.pad(x, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad))
            weight = np.pad(weight, (0, pad))
        return (
            torch.from_numpy(x.T.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(weight.astype(np.float32)),
        )


class TinyTCN(nn.Module):
    def __init__(self, channels: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden, dilation=2 ** (index % 5), dropout=dropout)
            for index in range(layers)
        )
        self.output = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input(x))
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def train_model(
    model: nn.Module,
    train_samples: list[StreamSample],
    args: argparse.Namespace,
    device: torch.device,
    fold_index: int,
    group: str,
    sample_seed: int,
    seed_index: int,
) -> None:
    dataset = ChunkDataset(
        train_samples,
        args.chunk_len,
        args.chunks_per_epoch,
        args.pos_sample_rate,
        sample_seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    iterator = range(args.epochs)
    if not args.quiet:
        iterator = tqdm(
            iterator,
            desc=f"fold {fold_index + 1} seed {seed_index + 1} {group[:18]}",
            leave=False,
        )
    for _ in iterator:
        running = 0.0
        for x, y, weight in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (loss * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
        if not args.quiet and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{running / max(1, len(loader)):.4f}")


@torch.inference_mode()
def predict_samples(
    model: nn.Module,
    samples: list[StreamSample],
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    outputs = []
    for sample in samples:
        x = torch.from_numpy(sample.x.T[None].astype(np.float32)).to(device)
        logits = model(x)
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy()[0].astype(np.float32))
    return outputs


def fit_standardizer(samples: list[StreamSample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([sample.x for sample in samples], axis=0)
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    threshold: float,
    nms_frames: int,
    cross_nms: int,
    count_mode: str,
    count_multiplier: float,
    pose_prior: float,
    snap_window: int = 0,
    stream_scores_by_key_group: dict[tuple[str, int], np.ndarray] | None = None,
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        selected = select_candidates(
            candidates_by_key[key],
            PoseHeuristicConfig(
                min_score=threshold,
                nms_frames=nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=cross_nms,
            ),
            estimate_count(video, train_videos, train_counts, gt_counts, count_mode, count_multiplier),
        )
        for candidate in selected:
            attrs = estimate_attrs(candidate, attr_priors)
            frame = snap_frame(candidate, video, snap_window, stream_scores_by_key_group)
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
    return rows


def snap_frame(
    candidate: PunchCandidate,
    video: dict[str, str],
    snap_window: int,
    stream_scores_by_key_group: dict[tuple[str, int], np.ndarray] | None,
) -> int:
    frame_count = int(video["frame_count"])
    frame = max(0, min(frame_count - 1, candidate.frame))
    if snap_window <= 0 or stream_scores_by_key_group is None:
        return frame
    scores = stream_scores_by_key_group.get((candidate.video_key, GROUP_INDEX[(candidate.fighter, candidate.hand)]))
    if scores is None or len(scores) == 0:
        return frame
    lo = max(0, frame - snap_window)
    hi = min(len(scores), frame + snap_window + 1)
    if hi <= lo:
        return frame
    return max(0, min(frame_count - 1, lo + int(np.argmax(scores[lo:hi]))))


def apply_pose_prior(candidate: PunchCandidate, alpha: float) -> PunchCandidate:
    pose_score = float(candidate.features.get("pose_score", candidate.score))
    score = candidate.score * (1.0 + alpha * np.log1p(max(0.0, pose_score)))
    return replace_score(candidate, float(score))


def replace_score(candidate: PunchCandidate, score: float) -> PunchCandidate:
    features = dict(candidate.features)
    features.setdefault("pose_score", candidate.score)
    return PunchCandidate(
        candidate.video_key,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
        score,
        features,
    )


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values, dtype=np.float32)])
    output = np.empty_like(values, dtype=np.float32)
    for index in range(len(values)):
        lo = max(0, index - window)
        hi = min(len(values), index + window + 1)
        output[index] = prefix[hi] - prefix[lo]
    return output


def other_hand_index(group_index: int) -> int:
    return group_index ^ 1


def other_fighter_index(group_index: int) -> int:
    return group_index ^ 2


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
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f} time={summary['time']:.6f} "
        f"fp={summary['fp_penalty']:.6f} wins={wins} n={n_rows}",
        flush=True,
    )


def print_best_detail(
    result: tuple[float, float, float, int, int, float, float, float, int, int, int, str, float],
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    gt: list[dict[str, str]],
    rows: list[dict[str, str]],
) -> None:
    (
        _score,
        _time,
        _fp,
        _wins,
        _n_pred,
        pose_prior,
        rgb_alpha,
        threshold,
        nms_frames,
        cross_nms,
        snap_window,
        count_mode,
        count_multiplier,
    ) = result
    score = score_predictions(gt, rows)
    selected_counts = Counter(row["video_key"] for row in rows)
    print(
        "best_detail: "
        f"score={score['macro_score']:.6f},pose_prior={pose_prior},rgb_alpha={rgb_alpha},threshold={threshold},"
        f"nms={nms_frames},cross_nms={cross_nms},count_mode={count_mode},"
        f"count_multiplier={count_multiplier},snap_window={snap_window},n={len(rows)}",
        flush=True,
    )
    print("best_roots: root,n_videos,score,time,fp,n_pred", flush=True)
    by_root: dict[str, list[str]] = defaultdict(list)
    for key in ready_keys:
        by_root[video_by_key[key]["data_root"]].append(key)
    for root, keys in sorted(by_root.items()):
        items = [score["by_video"][key] for key in keys]
        print(
            f"{root},{len(keys)},"
            f"{np.mean([item['final_score'] for item in items]):.6f},"
            f"{np.mean([item['score_time'] for item in items]):.6f},"
            f"{np.mean([item['fp_penalty'] for item in items]):.6f},"
            f"{sum(selected_counts[key] for key in keys)}",
            flush=True,
        )
    print("best_videos: video,root,score,time,fp,n_pred", flush=True)
    for key in ready_keys:
        item = score["by_video"][key]
        print(
            f"{key},{video_by_key[key]['data_root']},{item['final_score']:.6f},"
            f"{item['score_time']:.6f},{item['fp_penalty']:.6f},{selected_counts[key]}",
            flush=True,
        )


def build_result_rows(
    result: tuple[float, float, float, int, int, float, float, float, int, int, int, str, float],
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    scored_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    stream_scores_by_key_group: dict[tuple[str, int], np.ndarray],
    rgb_contact_by_key: dict[str, np.ndarray] | None,
) -> list[dict[str, str]]:
    (
        _score,
        _time,
        _fp,
        _wins,
        _n_pred,
        pose_prior,
        rgb_alpha,
        threshold,
        nms_frames,
        cross_nms,
        snap_window,
        count_mode,
        count_multiplier,
    ) = result
    reranked = {
        key: [apply_pose_prior(candidate, pose_prior) for candidate in candidates]
        for key, candidates in scored_by_key.items()
    }
    reranked = apply_rgb_contact_prior(reranked, rgb_contact_by_key, rgb_alpha)
    return build_rows(
        ready_keys,
        video_by_key,
        reranked,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        threshold,
        nms_frames,
        cross_nms,
        count_mode,
        count_multiplier,
        pose_prior=0.0,
        snap_window=snap_window,
        stream_scores_by_key_group=stream_scores_by_key_group,
    )


def compute_rgb_contact_probabilities(
    args: argparse.Namespace,
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    packs: dict[str, VideoPack],
    gt_by_key: dict[str, list[dict[str, str]]],
) -> dict[str, np.ndarray]:
    candidates = [candidate for key in ready_keys for candidate in packs[key].candidates]
    rows = rgb_candidate_feature_rows(candidates, video_by_key)
    labels, offsets = rgb_candidate_labels_offsets(candidates, gt_by_key, args.rgb_contact_label_window)
    groups = np.asarray([fight_group(video_by_key[candidate.video_key]) for candidate in candidates])
    rgb_args = SimpleNamespace(
        data_root=args.data_root,
        tracks_dir=args.tracks_dir,
        feature_cache=args.rgb_contact_feature_cache,
        model_name=args.rgb_contact_model_name,
        pretrained=True,
        device=args.device,
        image_size=args.rgb_contact_image_size,
        batch_size=args.rgb_contact_batch_size,
        clip_len=args.rgb_contact_clip_len,
        frame_stride=args.rgb_contact_frame_stride,
        crop_modes=args.rgb_contact_crop_modes,
        crop_expand=args.rgb_contact_crop_expand,
        decode_mode=args.rgb_contact_decode_mode,
        epochs=args.rgb_contact_epochs,
        head_batch_size=args.rgb_contact_head_batch_size,
        hidden=args.rgb_contact_hidden,
        dropout=args.rgb_contact_dropout,
        lr=args.rgb_contact_lr,
        weight_decay=args.rgb_contact_weight_decay,
        offset_loss_weight=args.rgb_contact_offset_loss_weight,
        offset_scale=args.rgb_contact_offset_scale,
        seed=args.seed + 777,
        quiet=args.quiet,
    )
    print(
        f"rgb_contact_labels n={len(candidates)} pos={int(labels.sum())} "
        f"pos_rate={float(labels.mean()):.4f} offset_mae={float(np.nanmean(np.abs(offsets))):.3f}",
        flush=True,
    )
    features = load_or_extract_rgb_contact_features(rgb_args, rows, video_by_key)
    if len(features) != len(candidates):
        raise ValueError(f"RGB feature count {len(features)} != candidate count {len(candidates)}")
    print(f"rgb_contact_features shape={features.shape}", flush=True)
    probabilities, _pred_offsets = fit_rgb_contact_heads(rgb_args, features, labels, offsets, groups)
    output = {}
    offset = 0
    print("rgb_contact_video,video_key,n,pos_rate,mean_p,pos_mean_p,neg_mean_p", flush=True)
    for key in ready_keys:
        n_items = len(packs[key].candidates)
        current = probabilities[offset : offset + n_items]
        y = labels[offset : offset + n_items]
        output[key] = current
        pos_p = current[y > 0.5]
        neg_p = current[y <= 0.5]
        print(
            f"rgb_contact_video,{key},{n_items},{float(y.mean()):.4f},"
            f"{float(current.mean()):.4f},"
            f"{float(pos_p.mean()) if len(pos_p) else 0.0:.4f},"
            f"{float(neg_p.mean()) if len(neg_p) else 0.0:.4f}",
            flush=True,
        )
        offset += n_items
    return output


def rgb_candidate_feature_rows(
    candidates: list[PunchCandidate],
    video_by_key: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        video = video_by_key[candidate.video_key]
        rows.append(
            {
                "id": str(index),
                "video_id": video["video_id"],
                "agn_index": video["agn_index"],
                "video_key": candidate.video_key,
                "frame": str(candidate.frame),
                "fighter": candidate.fighter,
                "hand": candidate.hand,
                "target": candidate.target,
                "punch_type": "",
                "effectiveness": "",
                "clear": "true",
            }
        )
    return rows


def rgb_candidate_labels_offsets(
    candidates: list[PunchCandidate],
    gt_by_key: dict[str, list[dict[str, str]]],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(candidates), dtype=np.float32)
    offsets = np.full(len(candidates), np.nan, dtype=np.float32)
    for index, candidate in enumerate(candidates):
        best: tuple[int, dict[str, str]] | None = None
        for row in gt_by_key.get(candidate.video_key, []):
            if row["fighter"] != candidate.fighter or row["hand"] != candidate.hand:
                continue
            distance = abs(candidate.frame - int(row["frame"]))
            if best is None or distance < best[0]:
                best = (distance, row)
        if best is not None and best[0] <= window:
            labels[index] = 1.0
            offsets[index] = float(int(best[1]["frame"]) - candidate.frame)
    return labels, offsets


def apply_rgb_contact_prior(
    candidates_by_key: dict[str, list[PunchCandidate]],
    rgb_contact_by_key: dict[str, np.ndarray] | None,
    alpha: float,
) -> dict[str, list[PunchCandidate]]:
    if rgb_contact_by_key is None or alpha == 0.0:
        return candidates_by_key
    all_probs = np.concatenate([rgb_contact_by_key[key] for key in sorted(rgb_contact_by_key)])
    mean_prob = float(all_probs.mean()) if len(all_probs) else 0.0
    output = {}
    for key, candidates in candidates_by_key.items():
        probs = rgb_contact_by_key[key]
        rows = []
        for candidate, probability in zip(candidates, probs):
            score = candidate.score * max(0.01, 1.0 + alpha * (float(probability) - mean_prob))
            features = dict(candidate.features)
            features["rgb_contact_prob"] = float(probability)
            rows.append(
                PunchCandidate(
                    candidate.video_key,
                    candidate.frame,
                    candidate.fighter,
                    candidate.hand,
                    candidate.target,
                    float(score),
                    features,
                )
            )
        output[key] = rows
    return output


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if torch.cuda.is_available() and visible == "1":
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(value)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not deterministic:
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
