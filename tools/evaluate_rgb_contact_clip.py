#!/usr/bin/env python3
"""Evaluate temporal RGB clip embeddings as a fixed-row contact/offset witness."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_rgb_event_filter import (
    crop_bounds,
    group_rows,
    image_to_tensor,
    load_model,
    load_track_records,
    pick_device,
    run_batch,
)
from tools.evaluate_rgb_timing_offset import fight_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--model-name", default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--crop-modes", default="union")
    parser.add_argument("--crop-expand", type=float, default=0.16)
    parser.add_argument("--decode-mode", choices=["sequential", "seek"], default="sequential")
    parser.add_argument("--video-decoder", choices=["opencv", "ffmpeg_cuda"], default="ffmpeg_cuda")
    parser.add_argument("--gpu-preprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--offset-loss-weight", type=float, default=0.15)
    parser.add_argument("--offset-scale", type=float, default=12.0)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70")
    parser.add_argument("--max-shifts", default="0,2,4,6,8")
    parser.add_argument("--shift-scales", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-threshold", type=float)
    parser.add_argument("--write-max-shift", type=int)
    parser.add_argument("--write-shift-scale", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    labels, target_offsets = matched_labels_offsets(gt_rows, pred_rows)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])

    features = load_or_extract_features(args, pred_rows, video_by_key)
    if len(features) != len(pred_rows):
        raise ValueError(f"feature count {len(features)} != prediction count {len(pred_rows)}")

    print(
        f"features shape={features.shape} model={args.model_name} "
        f"clip_len={args.clip_len} stride={args.frame_stride} crop_modes={args.crop_modes}",
        flush=True,
    )
    keep_prob, pred_offsets = fit_oof_temporal_heads(args, features, labels, target_offsets, groups)

    baseline = score_predictions(gt_rows, pred_rows)
    baseline_summary = score_summary(baseline)
    print(
        f"baseline score={baseline['macro_score']:.6f} "
        f"fighter={baseline_summary['fighter']:.6f} time={baseline_summary['time']:.6f} "
        f"fp={baseline_summary['fp']:.6f} n={len(pred_rows)} "
        f"pos_rate={float(labels.mean()):.4f} matched_offsets={int(np.isfinite(target_offsets).sum())}",
        flush=True,
    )
    print_probability_summary(pred_rows, labels, keep_prob, target_offsets, pred_offsets)

    results = evaluate_grid(args, pred_rows, gt_rows, video_by_key, keep_prob, pred_offsets, baseline)
    print("score,delta,fighter,time,fp,wins,n_rows,dropped,threshold,max_shift,shift_scale,mean_abs_shift")
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print_result(item, baseline["macro_score"])

    if args.write_oof_rows:
        chosen = choose_write_config(args, results)
        rows = apply_contact_rule(
            pred_rows,
            video_by_key,
            keep_prob,
            pred_offsets,
            chosen["threshold"],
            chosen["max_shift"],
            chosen["shift_scale"],
        )
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} threshold={chosen['threshold']} "
            f"max_shift={chosen['max_shift']} shift_scale={chosen['shift_scale']} n_rows={len(rows)}",
            flush=True,
        )
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def clip_offsets(clip_len: int, stride: int) -> list[int]:
    center = clip_len // 2
    return [(index - center) * stride for index in range(clip_len)]


def crop_modes(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    allowed = {"union", "full", "attacker", "opponent", "attacker_opponent"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown crop modes: {unknown}")
    return values or ["union"]


def load_or_extract_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> np.ndarray:
    if args.feature_cache.exists():
        data = np.load(args.feature_cache)
        return data["features"].astype(np.float32)
    return extract_features(args, pred_rows, video_by_key)


def extract_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> np.ndarray:
    cv2.setNumThreads(max(0, int(getattr(args, "cpu_threads", 1))))
    torch.set_num_threads(max(1, int(getattr(args, "cpu_threads", 1))))
    device = pick_device(args.device)
    print(f"extract_device={device}", flush=True)
    model, mean, std = load_model(args.model_name, args.pretrained, device)
    offsets = clip_offsets(args.clip_len, args.frame_stride)
    modes = crop_modes(args.crop_modes)
    per_event: list[list[np.ndarray | None]] = [[None] * (len(offsets) * len(modes)) for _ in pred_rows]
    row_to_index = {id(row): index for index, row in enumerate(pred_rows)}
    tensors: list[torch.Tensor] = []
    metas: list[tuple[int, int, int]] = []

    for key, rows in progress(group_rows(pred_rows).items(), "rgb-contact-features", args.quiet):
        video = video_by_key[key]
        frame_count = int(video["frame_count"])
        wanted_frames = {
            clamp_frame(int(row["frame"]) + offset, frame_count)
            for row in rows
            for offset in offsets
        }
        records = load_track_records(args.tracks_dir / f"{key}.jsonl", wanted_frames)
        video_path = args.data_root / video["video_path"]
        frame_jobs = build_frame_jobs(rows, row_to_index, offsets, frame_count)
        if args.video_decoder == "ffmpeg_cuda" and args.decode_mode == "sequential":
            extract_video_ffmpeg_cuda_sequential(
                args,
                video_path,
                int(video["width"]),
                int(video["height"]),
                frame_jobs,
                records,
                modes,
                mean,
                std,
                model,
                device,
                tensors,
                metas,
                per_event,
            )
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        try:
            if args.decode_mode == "seek":
                extract_video_seek(
                    args,
                    cap,
                    frame_jobs,
                    records,
                    modes,
                    mean,
                    std,
                    model,
                    device,
                    tensors,
                    metas,
                    per_event,
                )
            else:
                extract_video_sequential(
                    args,
                    cap,
                    frame_jobs,
                    records,
                    modes,
                    mean,
                    std,
                    model,
                    device,
                    tensors,
                    metas,
                    per_event,
                )
        finally:
            cap.release()
    if tensors:
        flush_sequence_batch(model, device, tensors, metas, per_event, len(modes))

    feature_dim = infer_feature_dim(per_event)
    output = np.zeros((len(pred_rows), len(offsets), feature_dim * len(modes)), dtype=np.float32)
    for event_index, items in enumerate(per_event):
        for time_index in range(len(offsets)):
            parts = []
            for mode_index in range(len(modes)):
                vector = items[time_index * len(modes) + mode_index]
                parts.append(vector if vector is not None else np.zeros(feature_dim, dtype=np.float32))
            output[event_index, time_index] = np.concatenate(parts).astype(np.float32)
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.feature_cache,
        features=output,
        clip_offsets=np.asarray(offsets, dtype=np.int16),
        crop_modes=np.asarray(modes),
        model_name=np.asarray([args.model_name]),
    )
    return output


def progress(items, desc: str, quiet: bool):
    return tqdm(list(items), desc=desc, leave=False, disable=quiet, mininterval=5.0)


FrameJob = tuple[int, int, dict[str, str]]


def build_frame_jobs(
    rows: list[dict[str, str]],
    row_to_index: dict[int, int],
    offsets: list[int],
    frame_count: int,
) -> dict[int, list[FrameJob]]:
    jobs: dict[int, list[FrameJob]] = defaultdict(list)
    for row in rows:
        event_index = row_to_index[id(row)]
        center = int(row["frame"])
        for time_index, offset in enumerate(offsets):
            frame = clamp_frame(center + offset, frame_count)
            jobs[frame].append((event_index, time_index, row))
    return jobs


def extract_video_seek(
    args: argparse.Namespace,
    cap: cv2.VideoCapture,
    frame_jobs: dict[int, list[FrameJob]],
    records: dict[int, dict[str, Any]],
    modes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, int]],
    per_event: list[list[np.ndarray | None]],
) -> None:
    for frame in sorted(frame_jobs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        append_frame_tensors(
            args,
            image if ok else None,
            records.get(frame),
            frame_jobs[frame],
            modes,
            mean,
            std,
            model,
            device,
            tensors,
            metas,
            per_event,
        )


def extract_video_sequential(
    args: argparse.Namespace,
    cap: cv2.VideoCapture,
    frame_jobs: dict[int, list[FrameJob]],
    records: dict[int, dict[str, Any]],
    modes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, int]],
    per_event: list[list[np.ndarray | None]],
) -> None:
    if not frame_jobs:
        return
    max_frame = max(frame_jobs)
    frame = 0
    while frame <= max_frame:
        ok, image = cap.read()
        if frame in frame_jobs:
            append_frame_tensors(
                args,
                image if ok else None,
                records.get(frame),
                frame_jobs[frame],
                modes,
                mean,
                std,
                model,
                device,
                tensors,
                metas,
                per_event,
            )
        if not ok and frame > max_frame:
            break
        frame += 1


def extract_video_ffmpeg_cuda_sequential(
    args: argparse.Namespace,
    video_path: Path,
    width: int,
    height: int,
    frame_jobs: dict[int, list[FrameJob]],
    records: dict[int, dict[str, Any]],
    modes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, int]],
    per_event: list[list[np.ndarray | None]],
) -> None:
    if not frame_jobs:
        return
    max_frame = max(frame_jobs)
    use_nv12_gpu = getattr(args, "gpu_preprocess", False) and device.type == "cuda"
    pix_fmt = "nv12" if use_nv12_gpu else "bgr24"
    frame_size = width * height * 3 // 2 if use_nv12_gpu else width * height * 3
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-hwaccel",
        "cuda",
        "-c:v",
        "hevc_cuvid",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")
    try:
        for frame in range(max_frame + 1):
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                image = None
                rgb_frame = None
            elif use_nv12_gpu:
                image = None
                rgb_frame = nv12_raw_to_rgb_tensor(raw, width, height, device)
            else:
                image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                rgb_frame = None
            if frame in frame_jobs:
                if rgb_frame is not None:
                    append_rgb_frame_tensors(
                        args,
                        rgb_frame,
                        (height, width, 3),
                        records.get(frame),
                        frame_jobs[frame],
                        modes,
                        mean,
                        std,
                        model,
                        device,
                        tensors,
                        metas,
                        per_event,
                    )
                else:
                    append_frame_tensors(
                        args,
                        image,
                        records.get(frame),
                        frame_jobs[frame],
                        modes,
                        mean,
                        std,
                        model,
                        device,
                        tensors,
                        metas,
                        per_event,
                    )
            if len(raw) != frame_size:
                break
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def nv12_raw_to_rgb_tensor(raw: bytes, width: int, height: int, device: torch.device) -> torch.Tensor:
    data = torch.frombuffer(raw, dtype=torch.uint8).to(device, non_blocking=True)
    y = data[: width * height].reshape(height, width).to(dtype=torch.float32)
    uv = data[width * height :].reshape(height // 2, width // 2, 2).permute(2, 0, 1).unsqueeze(0)
    uv = torch.nn.functional.interpolate(
        uv.to(dtype=torch.float32),
        size=(height, width),
        mode="nearest",
    ).squeeze(0)
    yy = torch.clamp(y - 16.0, min=0.0) * 1.164383
    u = uv[0] - 128.0
    v = uv[1] - 128.0
    red = torch.clamp(yy + 1.792741 * v, 0.0, 255.0)
    green = torch.clamp(yy - 0.213249 * u - 0.532909 * v, 0.0, 255.0)
    blue = torch.clamp(yy + 2.112402 * u, 0.0, 255.0)
    return torch.stack([red, green, blue], dim=0).div_(255.0)


def append_frame_tensors(
    args: argparse.Namespace,
    image: np.ndarray | None,
    record: dict[str, Any] | None,
    jobs: list[FrameJob],
    modes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, int]],
    per_event: list[list[np.ndarray | None]],
) -> None:
    gpu_image = None
    mean_tensor = None
    std_tensor = None
    if (
        getattr(args, "gpu_preprocess", False)
        and device.type == "cuda"
        and image is not None
        and image.flags["C_CONTIGUOUS"]
        and all(mode != "attacker_opponent" for mode in modes)
    ):
        gpu_image = torch.from_numpy(image).to(device, non_blocking=True)
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    for event_index, time_index, row in jobs:
        for mode_index, mode in enumerate(modes):
            if image is None:
                if getattr(args, "gpu_preprocess", False) and device.type == "cuda":
                    tensor = torch.zeros(3, args.image_size, args.image_size, device=device)
                else:
                    tensor = torch.zeros(3, args.image_size, args.image_size)
            elif gpu_image is not None and mean_tensor is not None and std_tensor is not None:
                tensor = image_to_tensor_from_gpu_frame(
                    gpu_image,
                    image.shape,
                    record,
                    row,
                    args.image_size,
                    args.crop_expand,
                    mean_tensor,
                    std_tensor,
                    mode,
                )
            else:
                tensor = image_to_tensor(
                    image,
                    record,
                    row,
                    args.image_size,
                    args.crop_expand,
                    mean,
                    std,
                    mode,
                )
            tensors.append(tensor)
            metas.append((event_index, time_index, mode_index))
            if len(tensors) >= args.batch_size:
                flush_sequence_batch(model, device, tensors, metas, per_event, len(modes))


def append_rgb_frame_tensors(
    args: argparse.Namespace,
    image: torch.Tensor,
    image_shape: tuple[int, ...],
    record: dict[str, Any] | None,
    jobs: list[FrameJob],
    modes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, int]],
    per_event: list[list[np.ndarray | None]],
) -> None:
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    for event_index, time_index, row in jobs:
        for mode_index, mode in enumerate(modes):
            if mode == "attacker_opponent":
                raise ValueError("ffmpeg_cuda NV12 GPU preprocessing does not support attacker_opponent crops")
            tensor = rgb_tensor_crop_from_gpu_frame(
                image,
                image_shape,
                record,
                row,
                args.image_size,
                args.crop_expand,
                mean_tensor,
                std_tensor,
                mode,
            )
            tensors.append(tensor)
            metas.append((event_index, time_index, mode_index))
            if len(tensors) >= args.batch_size:
                flush_sequence_batch(model, device, tensors, metas, per_event, len(modes))


def rgb_tensor_crop_from_gpu_frame(
    image: torch.Tensor,
    image_shape: tuple[int, ...],
    record: dict[str, Any] | None,
    row: dict[str, str],
    image_size: int,
    crop_expand: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    crop_mode: str,
) -> torch.Tensor:
    y1, y2, x1, x2 = crop_bounds(image_shape, record, row, crop_expand, crop_mode)
    crop = image[:, y1:y2, x1:x2]
    if crop.numel() == 0:
        crop = image
    tensor = torch.nn.functional.interpolate(
        crop.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    return (tensor - mean) / std


def image_to_tensor_from_gpu_frame(
    image: torch.Tensor,
    image_shape: tuple[int, ...],
    record: dict[str, Any] | None,
    row: dict[str, str],
    image_size: int,
    crop_expand: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    crop_mode: str,
) -> torch.Tensor:
    y1, y2, x1, x2 = crop_bounds(image_shape, record, row, crop_expand, crop_mode)
    crop = image[y1:y2, x1:x2]
    if crop.numel() == 0:
        crop = image
    tensor = crop.permute(2, 0, 1).flip(0).to(dtype=torch.float32).div_(255.0)
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    return (tensor - mean) / std


def flush_sequence_batch(
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, int]],
    per_event: list[list[np.ndarray | None]],
    n_modes: int,
) -> None:
    vectors = run_batch(model, device, tensors)
    for vector, (event_index, time_index, mode_index) in zip(vectors, metas):
        per_event[event_index][time_index * n_modes + mode_index] = vector
    metas.clear()


def infer_feature_dim(per_event: list[list[np.ndarray | None]]) -> int:
    for items in per_event:
        for vector in items:
            if vector is not None:
                return int(vector.shape[0])
    raise RuntimeError("No RGB features were extracted")


def matched_labels_offsets(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(pred_rows), dtype=np.float32)
    offsets = np.full(len(pred_rows), np.nan, dtype=np.float32)
    gt_by_key = group_rows(gt_rows)
    pred_by_key = group_rows(pred_rows)
    row_offset = 0
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            gt_frame = int(gt_by_key[key][match.gt_index]["frame"])
            pred_frame = int(rows[match.pred_index]["frame"])
            labels[row_offset + match.pred_index] = 1.0
            offsets[row_offset + match.pred_index] = float(gt_frame - pred_frame)
        row_offset += len(rows)
    return labels, offsets


class TemporalContactHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        pooled_dim = hidden * 2 * 3
        self.shared = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.keep = nn.Linear(hidden, 1)
        self.offset = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(self.input_norm(x))
        y, _ = self.gru(x)
        center = y[:, y.shape[1] // 2]
        mean = y.mean(dim=1)
        max_values = y.amax(dim=1)
        h = self.shared(torch.cat([center, mean, max_values], dim=1))
        return self.keep(h).squeeze(1), self.offset(h).squeeze(1)


def fit_oof_temporal_heads(
    args: argparse.Namespace,
    features: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = pick_device(args.device)
    output_prob = np.zeros(len(labels), dtype=np.float32)
    output_offset = np.zeros(len(labels), dtype=np.float32)
    for fold_index, group in enumerate(sorted(set(groups.tolist())), start=1):
        train = groups != group
        valid = groups == group
        if train.sum() < 20 or len(np.unique(labels[train])) < 2:
            output_prob[valid] = float(labels[train].mean()) if train.any() else 0.5
            output_offset[valid] = 0.0
            continue
        train_x, valid_x = normalize_fold(features[train], features[valid])
        fold_prob, fold_offset = train_fold(
            args,
            train_x,
            labels[train],
            offsets[train],
            valid_x,
            device,
            fold_index,
            group,
        )
        output_prob[valid] = fold_prob
        output_offset[valid] = fold_offset
    return output_prob, output_offset


def normalize_fold(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-4)
    return (train_x - mean) / std, (valid_x - mean) / std


def train_fold(
    args: argparse.Namespace,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_offsets: np.ndarray,
    valid_x: np.ndarray,
    device: torch.device,
    fold_index: int,
    group: str,
) -> tuple[np.ndarray, np.ndarray]:
    model = TemporalContactHead(train_x.shape[-1], args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    finite_offsets = np.isfinite(train_offsets)
    offset_targets = np.nan_to_num(train_offsets, nan=0.0).astype(np.float32) / args.offset_scale
    offset_mask = finite_offsets.astype(np.float32)
    pos = float(train_y.sum())
    neg = float(len(train_y) - train_y.sum())
    pos_weight = torch.tensor([max(0.25, min(4.0, neg / max(pos, 1.0)))], device=device)
    dataset = TensorDataset(
        torch.from_numpy(train_x.astype(np.float32)),
        torch.from_numpy(train_y.astype(np.float32)),
        torch.from_numpy(offset_targets.astype(np.float32)),
        torch.from_numpy(offset_mask.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + fold_index)
    loader = DataLoader(
        dataset,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    disable_bar = args.quiet
    for _epoch in tqdm(
        range(args.epochs),
        desc=f"contact-head fold={fold_index}",
        leave=False,
        disable=disable_bar,
        mininterval=5.0,
    ):
        model.train()
        for xb, yb, ob, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ob = ob.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            logits, pred_offset = model(xb)
            keep_loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            raw_offset_loss = nn.functional.smooth_l1_loss(pred_offset, ob, reduction="none")
            offset_loss = (raw_offset_loss * mb).sum() / torch.clamp(mb.sum(), min=1.0)
            loss = keep_loss + args.offset_loss_weight * offset_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        valid_tensor = torch.from_numpy(valid_x.astype(np.float32)).to(device)
        logits, pred_offset = model(valid_tensor)
        prob = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        offsets = (pred_offset.cpu().numpy().astype(np.float32) * args.offset_scale)
    print(
        f"fold_done={fold_index} group={group} train={len(train_y)} valid={len(valid_x)} "
        f"prob_mean={float(prob.mean()):.4f} offset_mae_train_target={float(np.nanmean(np.abs(train_offsets[finite_offsets]))):.3f}",
        flush=True,
    )
    return prob, offsets


def evaluate_grid(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    keep_prob: np.ndarray,
    pred_offsets: np.ndarray,
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for threshold in parse_floats(args.thresholds):
        for max_shift in parse_ints(args.max_shifts):
            for shift_scale in parse_floats(args.shift_scales):
                if max_shift == 0 and shift_scale != 0.0:
                    continue
                rows = apply_contact_rule(
                    pred_rows,
                    video_by_key,
                    keep_prob,
                    pred_offsets,
                    threshold,
                    max_shift,
                    shift_scale,
                )
                score = score_predictions(gt_rows, rows)
                summary = score_summary(score)
                mean_abs_shift = mean_shift(pred_rows, rows)
                results.append(
                    {
                        "score": score["macro_score"],
                        "fighter": summary["fighter"],
                        "time": summary["time"],
                        "fp": summary["fp"],
                        "wins": video_wins(score, baseline),
                        "n_rows": len(rows),
                        "dropped": len(pred_rows) - len(rows),
                        "threshold": threshold,
                        "max_shift": max_shift,
                        "shift_scale": shift_scale,
                        "mean_abs_shift": mean_abs_shift,
                    }
                )
    return results


def score_summary(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "fighter": float(np.mean([item["score_fighter"] for item in by_video.values()])),
        "time": float(np.mean([item["score_time"] for item in by_video.values()])),
        "fp": float(np.mean([item["fp_penalty"] for item in by_video.values()])),
    }


def apply_contact_rule(
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    keep_prob: np.ndarray,
    pred_offsets: np.ndarray,
    threshold: float,
    max_shift: int,
    shift_scale: float,
) -> list[dict[str, str]]:
    output = []
    for row, probability, offset in zip(rows, keep_prob, pred_offsets):
        if probability < threshold:
            continue
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        shift = int(round(float(np.clip(offset * shift_scale, -max_shift, max_shift))))
        frame_count = int(video_by_key[row["video_key"]]["frame_count"])
        item["frame"] = str(clamp_frame(int(row["frame"]) + shift, frame_count))
        output.append(item)
    return output


def mean_shift(before: list[dict[str, str]], after: list[dict[str, str]]) -> float:
    if not after:
        return 0.0
    before_by_id = {row["id"]: row for row in before}
    shifts = [abs(int(row["frame"]) - int(before_by_id[row["id"]]["frame"])) for row in after]
    return float(np.mean(shifts)) if shifts else 0.0


def choose_write_config(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    if args.write_threshold is not None:
        if args.write_max_shift is None or args.write_shift_scale is None:
            raise ValueError("--write-threshold requires --write-max-shift and --write-shift-scale")
        return {
            "threshold": args.write_threshold,
            "max_shift": args.write_max_shift,
            "shift_scale": args.write_shift_scale,
        }
    return max(results, key=lambda row: row["score"])


def print_result(item: dict[str, Any], baseline_score: float) -> None:
    print(
        f"{item['score']:.6f},{item['score'] - baseline_score:.6f},"
        f"{item['fighter']:.6f},{item['time']:.6f},{item['fp']:.6f},"
        f"{item['wins']},{item['n_rows']},{item['dropped']},{item['threshold']},"
        f"{item['max_shift']},{item['shift_scale']},{item['mean_abs_shift']:.3f}",
        flush=True,
    )


def print_probability_summary(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    keep_prob: np.ndarray,
    target_offsets: np.ndarray,
    pred_offsets: np.ndarray,
) -> None:
    print("video_key,n,pos_rate,mean_p,pos_mean_p,neg_mean_p,target_offset_mae,pred_offset_mae")
    start = 0
    for key, items in group_rows(rows).items():
        stop = start + len(items)
        y = labels[start:stop]
        p = keep_prob[start:stop]
        true_offset = target_offsets[start:stop]
        pred_offset = pred_offsets[start:stop]
        mask = np.isfinite(true_offset)
        pos_p = p[y > 0.5]
        neg_p = p[y <= 0.5]
        true_mae = float(np.mean(np.abs(true_offset[mask]))) if mask.any() else 0.0
        pred_mae = float(np.mean(np.abs(pred_offset[mask] - true_offset[mask]))) if mask.any() else 0.0
        print(
            f"{key},{len(items)},{float(y.mean()):.4f},{float(p.mean()):.4f},"
            f"{float(pos_p.mean()) if len(pos_p) else 0.0:.4f},"
            f"{float(neg_p.mean()) if len(neg_p) else 0.0:.4f},"
            f"{true_mae:.3f},{pred_mae:.3f}",
            flush=True,
        )
        start = stop


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def clamp_frame(frame: int, frame_count: int) -> int:
    return max(0, min(frame_count - 1, int(frame)))


if __name__ == "__main__":
    raise SystemExit(main())
