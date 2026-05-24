#!/usr/bin/env python3
"""Evaluate frozen DINO patch-token ROI features for fixed-row attributes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    PoseHeuristicConfig,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)
from tools.evaluate_rgb_contact_clip import (
    clamp_frame,
    clip_offsets,
    nv12_raw_to_rgb_tensor,
)
from tools.evaluate_rgb_event_filter import crop_bounds, group_rows
from tools.evaluate_rgb_fixed_row_attribute_model import (
    ATTR_COLUMNS,
    apply_attribute_margin,
    apply_predictions,
    attr_summary,
    matched_attribute_labels,
    oof_attribute_predictions,
    row_context_features,
)
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_rgb_timing_offset import fight_group


REGION_NAMES = ["wrist", "head", "body", "target", "midpoint"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--crop-expand", type=float, default=0.22)
    parser.add_argument("--roi-radius", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--train-columns", default="punch_type,effectiveness,target")
    parser.add_argument("--margin-columns", default="punch_type,effectiveness,target")
    parser.add_argument("--effectiveness-margins", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--head-type", choices=["torch_linear", "torch_mlp", "logreg"], default="torch_linear")
    parser.add_argument("--pca-components", type=int, default=160)
    parser.add_argument("--logreg-c", type=float, default=0.25)
    parser.add_argument("--torch-hidden-dim", type=int, default=192)
    parser.add_argument("--torch-depth", type=int, default=1)
    parser.add_argument("--torch-dropout", type=float, default=0.10)
    parser.add_argument("--torch-epochs", type=int, default=140)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-lr", type=float, default=2e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-3)
    parser.add_argument("--torch-seed", type=int, default=20260522)
    parser.add_argument("--torch-class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--torch-pca-components", type=int, default=160)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-margin-column", choices=ATTR_COLUMNS, default="punch_type")
    parser.add_argument("--write-effectiveness-margin", type=float)
    parser.add_argument(
        "--write-variant",
        choices=["punch_type", "effectiveness", "ptype_eff", "hand_target", "all_attrs"],
        default="punch_type",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.cpu_threads)))

    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    missing = sorted(set(keys) - set(video_by_key))
    if missing:
        raise RuntimeError(f"Prediction keys are not train videos: {missing}")
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]

    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, len(pred_rows), 0)

    features = load_or_extract_features(args, pred_rows, video_by_key)
    print(
        f"dino_roi_features={features.shape} rows={len(pred_rows)} "
        f"model={args.model_name} cache={args.feature_cache}",
        flush=True,
    )
    x = build_event_features(features, pred_rows, video_by_key)
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    matched = np.ones(len(pred_rows), dtype=bool)
    train_columns = split_columns(args.train_columns)
    margin_columns = split_columns(args.margin_columns)
    print(
        f"x_shape={x.shape} train_columns={','.join(train_columns)} "
        f"head={args.head_type} torch_pca={args.torch_pca_components}",
        flush=True,
    )
    predictions, probabilities = oof_attribute_predictions(args, x, labels, groups, pred_rows, matched, train_columns)
    variants = {
        "punch_type": ["punch_type"],
        "effectiveness": ["effectiveness"],
        "ptype_eff": ["punch_type", "effectiveness"],
        "hand_target": ["hand", "target"],
        "all_attrs": ATTR_COLUMNS,
    }

    rows_by_variant = {}
    print("variant,score,delta,type,effectiveness,hand,target,wins,n_changed")
    for name, columns in variants.items():
        rows, changed = apply_predictions(pred_rows, predictions, columns)
        rows_by_variant[name] = rows
        print_variant(name, rows, gt_rows, baseline, changed)
    for column in margin_columns:
        for margin in parse_floats(args.effectiveness_margins):
            name = f"{column}_margin_{margin:g}"
            rows, changed = apply_attribute_margin(pred_rows, predictions, probabilities, column, margin)
            rows_by_variant[name] = rows
            print_variant(name, rows, gt_rows, baseline, changed)

    if args.write_oof_rows:
        variant_name = args.write_variant
        if args.write_effectiveness_margin is not None:
            variant_name = f"{args.write_margin_column}_margin_{args.write_effectiveness_margin:g}"
        if variant_name not in rows_by_variant:
            raise KeyError(f"variant {variant_name} was not evaluated")
        write_csv_rows(args.write_oof_rows, rows_by_variant[variant_name], SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} variant={variant_name}", flush=True)
    return 0


def load_or_extract_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> np.ndarray:
    if args.feature_cache.exists():
        data = np.load(args.feature_cache)
        return data["features"].astype(np.float32)
    features = extract_features(args, pred_rows, video_by_key)
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.feature_cache,
        features=features,
        clip_offsets=np.asarray(clip_offsets(args.clip_len, args.frame_stride), dtype=np.int16),
        region_names=np.asarray(REGION_NAMES),
        model_name=np.asarray([args.model_name]),
        image_size=np.asarray([args.image_size], dtype=np.int16),
    )
    return features


def extract_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> np.ndarray:
    device = pick_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("DINO ROI extraction is intended for CUDA/NVDEC; set CUDA_VISIBLE_DEVICES=1 --device cuda")
    model = load_dense_model(args.model_name, args.pretrained, device)
    mean, std = model_normalization(model, device)
    patch_grid = int(args.image_size // patch_size(model))
    offsets = clip_offsets(args.clip_len, args.frame_stride)
    per_event: list[list[np.ndarray | None]] = [[None] * len(offsets) for _ in pred_rows]
    row_to_index = {id(row): index for index, row in enumerate(pred_rows)}
    tensors: list[torch.Tensor] = []
    metas: list[tuple[int, int, list[np.ndarray], np.ndarray]] = []

    for key, rows in progress(group_rows(pred_rows).items(), "dino-roi-features", args.quiet):
        video = video_by_key[key]
        frame_count = int(video["frame_count"])
        wanted_frames = {
            clamp_frame(int(row["frame"]) + offset, frame_count)
            for row in rows
            for offset in offsets
        }
        records = load_track_records(args.tracks_dir / f"{key}.jsonl", wanted_frames)
        frame_jobs = build_frame_jobs(rows, row_to_index, offsets, frame_count)
        extract_video_ffmpeg_cuda(
            args,
            args.data_root / video["video_path"],
            int(video["width"]),
            int(video["height"]),
            frame_jobs,
            records,
            model,
            mean,
            std,
            patch_grid,
            device,
            tensors,
            metas,
            per_event,
        )
    if tensors:
        flush_dense_batch(model, device, tensors, metas, per_event, patch_grid)
    feature_dim = infer_feature_dim(per_event)
    output = np.zeros((len(pred_rows), len(offsets), feature_dim), dtype=np.float32)
    for event_index, items in enumerate(per_event):
        for time_index, vector in enumerate(items):
            if vector is not None:
                output[event_index, time_index] = vector
    return output


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


def extract_video_ffmpeg_cuda(
    args: argparse.Namespace,
    video_path: Path,
    width: int,
    height: int,
    frame_jobs: dict[int, list[FrameJob]],
    records: dict[int, dict[str, Any]],
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    patch_grid: int,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, list[np.ndarray], np.ndarray]],
    per_event: list[list[np.ndarray | None]],
) -> None:
    if not frame_jobs:
        return
    max_frame = max(frame_jobs)
    frame_size = width * height * 3 // 2
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
        "nv12",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")
    try:
        for frame in range(max_frame + 1):
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            if frame not in frame_jobs:
                continue
            image = nv12_raw_to_rgb_tensor(raw, width, height, device)
            for event_index, time_index, row in frame_jobs[frame]:
                record = records.get(frame)
                tensor, roi_indices, geom = tensor_and_roi_meta(
                    args,
                    image,
                    (height, width, 3),
                    record,
                    row,
                    mean,
                    std,
                    patch_grid,
                )
                tensors.append(tensor)
                metas.append((event_index, time_index, roi_indices, geom))
                if len(tensors) >= args.batch_size:
                    flush_dense_batch(model, device, tensors, metas, per_event, patch_grid)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def tensor_and_roi_meta(
    args: argparse.Namespace,
    image: torch.Tensor,
    image_shape: tuple[int, ...],
    record: dict[str, Any] | None,
    row: dict[str, str],
    mean: torch.Tensor,
    std: torch.Tensor,
    patch_grid: int,
) -> tuple[torch.Tensor, list[np.ndarray], np.ndarray]:
    y1, y2, x1, x2 = crop_bounds(image_shape, record, row, args.crop_expand, "union")
    crop = image[:, y1:y2, x1:x2]
    if crop.numel() == 0:
        crop = image
        y1, y2, x1, x2 = 0, image_shape[0], 0, image_shape[1]
    tensor = torch.nn.functional.interpolate(
        crop.unsqueeze(0),
        size=(args.image_size, args.image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    tensor = (tensor - mean) / std
    roi_points, geom = pose_roi_points(record, row, (x1, y1, x2, y2), args.image_size)
    roi_indices = [point_to_patch_indices(point, patch_grid, args.image_size, args.roi_radius) for point in roi_points]
    return tensor, roi_indices, geom


def flush_dense_batch(
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int, list[np.ndarray], np.ndarray]],
    per_event: list[list[np.ndarray | None]],
    patch_grid: int,
) -> None:
    x = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", enabled=True):
            tokens = dense_tokens(model, x)
    cls = torch.nn.functional.normalize(tokens[:, 0].float(), dim=1)
    patches = torch.nn.functional.normalize(tokens[:, 1:].float(), dim=2).reshape(
        len(tensors), patch_grid * patch_grid, -1
    )
    global_patch = torch.nn.functional.normalize(patches.mean(dim=1), dim=1)
    for sample_index, (event_index, time_index, roi_indices, geom) in enumerate(metas):
        region_vectors = [
            pool_region(patches[sample_index], indices)
            for indices in roi_indices
        ]
        wrist, head, body, target, midpoint = region_vectors
        alt_target = body if np.allclose(geom[0], 1.0) else head
        parts = [
            cls[sample_index],
            global_patch[sample_index],
            wrist,
            head,
            body,
            target,
            midpoint,
            wrist - target,
            torch.abs(wrist - target),
            wrist - alt_target,
            torch.abs(wrist - alt_target),
        ]
        visual = torch.cat(parts).detach().cpu().numpy().astype(np.float32)
        per_event[event_index][time_index] = np.concatenate([visual, geom.astype(np.float32)])
    tensors.clear()
    metas.clear()


def dense_tokens(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model.forward_features(x)
    if isinstance(y, dict):
        if "x_norm_clstoken" in y and "x_norm_patchtokens" in y:
            return torch.cat([y["x_norm_clstoken"].unsqueeze(1), y["x_norm_patchtokens"]], dim=1)
        for key in ["x", "last_hidden_state"]:
            if key in y:
                return y[key]
        raise KeyError(f"Unsupported forward_features keys: {sorted(y)}")
    return y


def pool_region(patches: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    if indices.size == 0:
        return torch.nn.functional.normalize(patches.mean(dim=0), dim=0)
    index = torch.as_tensor(indices, dtype=torch.long, device=patches.device)
    return torch.nn.functional.normalize(patches.index_select(0, index).mean(dim=0), dim=0)


def pose_roi_points(
    record: dict[str, Any] | None,
    row: dict[str, str],
    bounds: tuple[int, int, int, int],
    image_size: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    attacker_role = row["fighter"]
    opponent_role = "blue" if attacker_role == "red" else "red"
    fighters = (record or {}).get("fighters", {})
    attacker = fighters.get(attacker_role, {})
    opponent = fighters.get(opponent_role, {})
    wrist_index = L_WRIST if row.get("hand") == "left" else R_WRIST
    wrist = keypoint(attacker, wrist_index, 0.15)
    if wrist is None:
        wrist = bbox_center(attacker)
    head = target_point(opponent, "head")
    if head is None:
        head = bbox_center(opponent)
    body = target_point(opponent, "body")
    if body is None:
        body = bbox_center(opponent)
    if wrist is None:
        wrist = np.asarray([(bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5], dtype=np.float32)
    if head is None:
        head = wrist.copy()
    if body is None:
        body = head.copy()
    target = head if row.get("target") == "head" else body
    midpoint = (wrist + target) * 0.5
    points = [wrist, head, body, target, midpoint]
    mapped = [map_point(point, bounds, image_size) for point in points]

    scale = max(1.0, float(max(bounds[2] - bounds[0], bounds[3] - bounds[1])))
    wrist_head = float(np.linalg.norm(wrist - head) / scale)
    wrist_body = float(np.linalg.norm(wrist - body) / scale)
    conf = keypoint_conf(attacker, wrist_index)
    geom = np.asarray(
        [
            1.0 if row.get("target") == "head" else 0.0,
            1.0 if row.get("hand") == "left" else 0.0,
            1.0 if attacker_role == "red" else 0.0,
            wrist_head,
            wrist_body,
            min(wrist_head, wrist_body),
            conf,
            float(attacker.get(f"score_{attacker_role}", 0.0) or 0.0),
            float(opponent.get(f"score_{opponent_role}", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )
    return mapped, geom


def map_point(point: np.ndarray, bounds: tuple[int, int, int, int], image_size: int) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    x = (float(point[0]) - x1) / width * image_size
    y = (float(point[1]) - y1) / height * image_size
    return np.asarray([np.clip(x, 0.0, image_size - 1.0), np.clip(y, 0.0, image_size - 1.0)], dtype=np.float32)


def point_to_patch_indices(point: np.ndarray, patch_grid: int, image_size: int, radius: int) -> np.ndarray:
    cell = max(1.0, float(image_size) / patch_grid)
    col = int(np.clip(np.floor(point[0] / cell), 0, patch_grid - 1))
    row = int(np.clip(np.floor(point[1] / cell), 0, patch_grid - 1))
    indices = []
    for yy in range(max(0, row - radius), min(patch_grid, row + radius + 1)):
        for xx in range(max(0, col - radius), min(patch_grid, col + radius + 1)):
            indices.append(yy * patch_grid + xx)
    return np.asarray(indices, dtype=np.int64)


def build_event_features(
    features: np.ndarray,
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> np.ndarray:
    center_index = features.shape[1] // 2
    mean = features.mean(axis=1)
    std = features.std(axis=1)
    center = features[:, center_index]
    delta = features[:, -1] - features[:, 0]
    context = np.stack([row_context_features(row, video_by_key[row["video_key"]]) for row in rows]).astype(np.float32)
    return np.concatenate([center, mean, std, delta, center - mean, context], axis=1).astype(np.float32)


def load_dense_model(model_name: str, pretrained: bool, device: torch.device) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.eval().to(device)
    return model


def model_normalization(model: torch.nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = getattr(model, "default_cfg", {}) or {}
    mean = torch.as_tensor(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32, device=device)[:, None, None]
    std = torch.as_tensor(cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32, device=device)[:, None, None]
    return mean, std


def patch_size(model: torch.nn.Module) -> int:
    value = getattr(getattr(model, "patch_embed", None), "patch_size", (14, 14))
    return int(value[0] if isinstance(value, tuple) else value)


def load_track_records(path: Path, frames: set[int]) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as fh:
        import json

        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame"])
            if frame in frames:
                records[frame] = record
    return records


def keypoint(fighter: dict[str, Any], index: int, min_conf: float) -> np.ndarray | None:
    keypoints = fighter.get("keypoints") or []
    if index >= len(keypoints):
        return None
    kp = keypoints[index]
    if len(kp) < 3 or float(kp[2]) < min_conf:
        return None
    return np.asarray([float(kp[0]), float(kp[1])], dtype=np.float32)


def keypoint_conf(fighter: dict[str, Any], index: int) -> float:
    keypoints = fighter.get("keypoints") or []
    if index >= len(keypoints) or len(keypoints[index]) < 3:
        return 0.0
    return float(keypoints[index][2])


def target_point(fighter: dict[str, Any], target: str) -> np.ndarray | None:
    config = PoseHeuristicConfig()
    if target == "head":
        points = [keypoint(fighter, index, config.kp_conf * 0.8) for index in [0, 1, 2, 3, 4]]
        visible = [point for point in points if point is not None]
        if visible:
            return np.mean(visible, axis=0).astype(np.float32)
        bbox = fighter.get("bbox")
        if bbox:
            return np.asarray([(bbox[0] + bbox[2]) * 0.5, bbox[1] + (bbox[3] - bbox[1]) * 0.12], dtype=np.float32)
        return None

    points = [keypoint(fighter, index, config.kp_conf * 0.8) for index in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
    visible = [point for point in points if point is not None]
    if visible:
        return np.mean(visible, axis=0).astype(np.float32)
    bbox = fighter.get("bbox")
    if bbox:
        return np.asarray([(bbox[0] + bbox[2]) * 0.5, bbox[1] + (bbox[3] - bbox[1]) * 0.52], dtype=np.float32)
    return None


def bbox_center(fighter: dict[str, Any]) -> np.ndarray | None:
    bbox = fighter.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)


def infer_feature_dim(per_event: list[list[np.ndarray | None]]) -> int:
    for items in per_event:
        for vector in items:
            if vector is not None:
                return int(vector.shape[0])
    raise RuntimeError("No DINO ROI features were extracted")


def print_score(label: str, score: dict[str, Any], n_rows: int, wins: int) -> None:
    summary = attr_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},"
        f"type={summary['punch_type']:.6f},eff={summary['effectiveness']:.6f},"
        f"hand={summary['hand']:.6f},target={summary['target']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def print_variant(
    name: str,
    rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    baseline: dict[str, Any],
    changed: int,
) -> None:
    score = score_predictions(gt_rows, rows)
    summary = attr_summary(score)
    print(
        f"{name},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:.6f},"
        f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
        f"{summary['hand']:.6f},{summary['target']:.6f},"
        f"{video_wins(score, baseline)},{changed}",
        flush=True,
    )


def split_columns(text: str) -> list[str]:
    columns = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(columns) - set(ATTR_COLUMNS))
    if unknown:
        raise ValueError(f"unknown columns: {unknown}")
    return columns


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def pick_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def progress(items, desc: str, quiet: bool):
    return tqdm(list(items), desc=desc, leave=False, disable=quiet, mininterval=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
