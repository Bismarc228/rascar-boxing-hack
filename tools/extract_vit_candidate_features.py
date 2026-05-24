#!/usr/bin/env python3
"""Extract local ViT/CLIP/VideoMAE features for fixed prediction rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel, VideoMAEModel

from rascar_boxing.io import as_int, read_csv_rows


BACKBONES = {
    "vit_in21k": ("image_auto", "google__vit-base-patch16-224-in21k"),
    "dino_vitb16": ("image_auto", "facebook__dino-vitb16"),
    "clip_vitb16": ("clip", "openai__clip-vit-base-patch16"),
    "videomae_k400": ("videomae", "MCG-NJU__videomae-base-finetuned-kinetics"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("data/processed/model_zoo/hf"))
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="clip_vitb16")
    parser.add_argument("--crop-modes", default="union")
    parser.add_argument("--frame-offsets", default="-4,0,4")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--crop-expand", type=float, default=0.16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-sequential-gap", type=int, default=8)
    parser.add_argument("--preprocess-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--log-batches", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    if args.max_rows:
        pred_rows = pred_rows[: args.max_rows]
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    missing = sorted({row["video_key"] for row in pred_rows} - set(video_by_key))
    if missing:
        raise ValueError(f"Prediction rows are not train videos: {missing}")

    extractor = FeatureExtractor(args)
    features = extractor.extract(pred_rows, video_by_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features.astype(np.float32),
        backbone=np.asarray([args.backbone]),
        crop_modes=np.asarray(parse_list(args.crop_modes)),
        frame_offsets=np.asarray(parse_ints(args.frame_offsets), dtype=np.int16),
        clip_len=np.asarray([args.clip_len], dtype=np.int16),
        frame_stride=np.asarray([args.frame_stride], dtype=np.int16),
        preprocess_device=np.asarray([args.preprocess_device]),
        max_sequential_gap=np.asarray([args.max_sequential_gap], dtype=np.int16),
    )
    write_index(args.index_output, pred_rows)
    print(f"wrote_features={args.output} shape={features.shape}")
    print(f"wrote_index={args.index_output} rows={len(pred_rows)}")
    return 0


class FeatureExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.enforce_gpu_visibility()
        kind, dirname = BACKBONES[args.backbone]
        self.kind = kind
        self.model_path = args.model_root / dirname
        self.processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.size = processor_size(self.processor, args.image_size)
        self.mean = np.asarray(getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(getattr(self.processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        if args.preprocess_device == "cuda" and self.device.type != "cuda":
            raise ValueError("--preprocess-device cuda requires a CUDA model device")
        self.mean_tensor = torch.as_tensor(self.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.as_tensor(self.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        if kind == "clip":
            self.model = CLIPVisionModel.from_pretrained(self.model_path, local_files_only=True).eval().to(self.device)
        elif kind == "videomae":
            self.model = VideoMAEModel.from_pretrained(self.model_path, local_files_only=True).eval().to(self.device)
        else:
            self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True).eval().to(self.device)
        self.batch_count = 0
        if not args.quiet:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
            print(
                f"extractor_device={self.device} device_name={device_name} "
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
                f"preprocess_device={args.preprocess_device} "
                f"backbone={args.backbone} model_path={self.model_path}",
                flush=True,
            )

    def enforce_gpu_visibility(self) -> None:
        if self.device.type != "cuda":
            return
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if tokens != ["1"]:
            raise RuntimeError(
                "CUDA ViT extraction must run with CUDA_VISIBLE_DEVICES=1 "
                f"to avoid physical GPU 0; got CUDA_VISIBLE_DEVICES={visible!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if self.device.index not in (None, 0):
            raise RuntimeError(
                "Use --device cuda or --device cuda:0 with CUDA_VISIBLE_DEVICES=1; "
                f"got {self.device}"
            )
        self.device = torch.device("cuda:0")

    def extract(
        self,
        rows: list[dict[str, str]],
        video_by_key: dict[str, dict[str, str]],
    ) -> np.ndarray:
        modes = parse_list(self.args.crop_modes)
        if self.kind == "videomae":
            return self.extract_video_features(rows, video_by_key, modes)
        return self.extract_image_features(rows, video_by_key, modes)

    def extract_image_features(
        self,
        rows: list[dict[str, str]],
        video_by_key: dict[str, dict[str, str]],
        modes: list[str],
    ) -> np.ndarray:
        offsets = parse_ints(self.args.frame_offsets)
        output: list[list[np.ndarray | None]] = [[None] * (len(offsets) * len(modes)) for _ in rows]
        batch: list[torch.Tensor | np.ndarray] = []
        metas: list[tuple[int, int, int]] = []
        grouped = group_rows(rows)
        row_index = {id(row): index for index, row in enumerate(rows)}
        for key, group in progress(grouped.items(), "vit-image-features", self.args.quiet):
            video = video_by_key[key]
            frame_count = as_int(video["frame_count"], "frame_count")
            jobs_by_frame: dict[int, list[tuple[int, int, int, dict[str, str], str]]] = defaultdict(list)
            for row in group:
                event_index = row_index[id(row)]
                center = as_int(row["frame"], "frame")
                for time_index, offset in enumerate(offsets):
                    frame = clamp_frame(center + offset, frame_count)
                    for mode_index, mode in enumerate(modes):
                        jobs_by_frame[frame].append((event_index, time_index, mode_index, row, mode))
            records = load_track_records(self.args.tracks_dir / f"{key}.jsonl", set(jobs_by_frame))
            video_path = self.args.data_root / video["video_path"]
            for frame, image in iter_wanted_frames(video_path, set(jobs_by_frame), self.args.max_sequential_gap):
                record = records.get(frame)
                for event_index, time_index, mode_index, row, mode in jobs_by_frame[frame]:
                    crop = crop_image(image, record, row, self.args.crop_expand, mode)
                    batch.append(self.preprocess_crop(crop))
                    metas.append((event_index, time_index, mode_index))
                    if len(batch) >= self.args.batch_size:
                        self.flush_image_batch(batch, metas, output, len(modes))
        if batch:
            self.flush_image_batch(batch, metas, output, len(modes))
        feature_dim = infer_feature_dim(output)
        array = np.zeros((len(rows), len(offsets), feature_dim * len(modes)), dtype=np.float32)
        for row_idx, items in enumerate(output):
            for time_index in range(len(offsets)):
                vectors = []
                for mode_index in range(len(modes)):
                    vector = items[time_index * len(modes) + mode_index]
                    vectors.append(vector if vector is not None else np.zeros(feature_dim, dtype=np.float32))
                array[row_idx, time_index] = np.concatenate(vectors)
        return array

    def flush_image_batch(
        self,
        batch: list[torch.Tensor | np.ndarray],
        metas: list[tuple[int, int, int]],
        output: list[list[np.ndarray | None]],
        n_modes: int,
    ) -> None:
        batch_size = len(batch)
        started = time.perf_counter()
        x = self.make_image_batch(batch)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            if self.kind == "clip":
                y = self.model(pixel_values=x).pooler_output
            else:
                y = self.model(pixel_values=x).last_hidden_state[:, 0]
        y = torch.nn.functional.normalize(y.float(), dim=1).cpu().numpy().astype(np.float32)
        self.batch_count += 1
        if self.args.log_batches:
            print(
                f"flush={self.batch_count} kind=image batch={batch_size} sec={time.perf_counter() - started:.3f} "
                f"device={self.device}",
                flush=True,
            )
        for vector, (event_index, time_index, mode_index) in zip(y, metas):
            output[event_index][time_index * n_modes + mode_index] = vector
        batch.clear()
        metas.clear()

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor | np.ndarray:
        if self.args.preprocess_device == "cuda":
            return as_uint8_hwc(crop)
        return to_tensor(crop, self.size, self.mean, self.std)

    def preprocess_clip(self, slots: list[torch.Tensor | np.ndarray | None]) -> torch.Tensor | list[np.ndarray]:
        items = [item for item in slots if item is not None]
        if self.args.preprocess_device == "cuda":
            raw = [item for item in items if isinstance(item, np.ndarray)]
            if len(raw) != len(items):
                raise RuntimeError("mixed CPU/CUDA preprocess items in video clip")
            return raw
        tensors = [item for item in items if isinstance(item, torch.Tensor)]
        if len(tensors) != len(items):
            raise RuntimeError("mixed CPU/CUDA preprocess items in video clip")
        return torch.stack(tensors)

    def make_image_batch(self, batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor:
        if self.args.preprocess_device == "cuda":
            raw = [item for item in batch if isinstance(item, np.ndarray)]
            if len(raw) != len(batch):
                raise RuntimeError("mixed CPU/CUDA preprocess items in image batch")
            return self.gpu_preprocess_batch(raw)
        tensors = [item for item in batch if isinstance(item, torch.Tensor)]
        if len(tensors) != len(batch):
            raise RuntimeError("mixed CPU/CUDA preprocess items in image batch")
        return torch.stack(tensors).to(self.device, non_blocking=True)

    def make_video_batch(self, batch: list[torch.Tensor | list[np.ndarray]]) -> torch.Tensor:
        if self.args.preprocess_device == "cuda":
            clips = [item for item in batch if isinstance(item, list)]
            if len(clips) != len(batch):
                raise RuntimeError("mixed CPU/CUDA preprocess items in video batch")
            clip_len = len(clips[0])
            if any(len(clip) != clip_len for clip in clips):
                raise RuntimeError("inconsistent clip lengths in video batch")
            flat = [crop for clip in clips for crop in clip]
            x = self.gpu_preprocess_batch(flat)
            return x.view(len(clips), clip_len, 3, self.size, self.size)
        tensors = [item for item in batch if isinstance(item, torch.Tensor)]
        if len(tensors) != len(batch):
            raise RuntimeError("mixed CPU/CUDA preprocess items in video batch")
        return torch.stack(tensors).to(self.device, non_blocking=True)

    def gpu_preprocess_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        if not crops:
            raise RuntimeError("empty CUDA preprocess batch")
        tensors = []
        for crop in crops:
            tensor = torch.from_numpy(crop).to(self.device, non_blocking=True)
            tensor = tensor[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(
                tensor.float(),
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )
            tensors.append(tensor)
        x = torch.cat(tensors, dim=0).div_(255.0)
        return (x - self.mean_tensor) / self.std_tensor

    def extract_video_features(
        self,
        rows: list[dict[str, str]],
        video_by_key: dict[str, dict[str, str]],
        modes: list[str],
    ) -> np.ndarray:
        output: list[list[np.ndarray | None]] = [[None] * len(modes) for _ in rows]
        batch: list[torch.Tensor | list[np.ndarray]] = []
        metas: list[tuple[int, int]] = []
        offsets = centered_offsets(self.args.clip_len, self.args.frame_stride)
        grouped = group_rows(rows)
        row_index = {id(row): index for index, row in enumerate(rows)}
        for key, group in progress(grouped.items(), "vit-video-features", self.args.quiet):
            video = video_by_key[key]
            frame_count = as_int(video["frame_count"], "frame_count")
            jobs_by_frame: dict[int, list[tuple[tuple[int, int], int, dict[str, str], str]]] = defaultdict(list)
            clip_slots: dict[tuple[int, int], list[torch.Tensor | np.ndarray | None]] = {}
            remaining: dict[tuple[int, int], int] = {}
            for row in group:
                event_index = row_index[id(row)]
                center = as_int(row["frame"], "frame")
                for mode_index, mode in enumerate(modes):
                    clip_key = (event_index, mode_index)
                    clip_slots[clip_key] = [None] * len(offsets)
                    remaining[clip_key] = len(offsets)
                    for offset_index, offset in enumerate(offsets):
                        frame = clamp_frame(center + offset, frame_count)
                        jobs_by_frame[frame].append((clip_key, offset_index, row, mode))
            records = load_track_records(self.args.tracks_dir / f"{key}.jsonl", set(jobs_by_frame))
            video_path = self.args.data_root / video["video_path"]
            for frame, image in iter_wanted_frames(video_path, set(jobs_by_frame), self.args.max_sequential_gap):
                record = records.get(frame)
                for clip_key, offset_index, row, mode in jobs_by_frame[frame]:
                    crop = crop_image(image, record, row, self.args.crop_expand, mode)
                    slots = clip_slots[clip_key]
                    if slots[offset_index] is None:
                        remaining[clip_key] -= 1
                    slots[offset_index] = self.preprocess_crop(crop)
                    if remaining[clip_key] == 0:
                        batch.append(self.preprocess_clip(slots))
                        metas.append(clip_key)
                        del clip_slots[clip_key]
                        del remaining[clip_key]
                        if len(batch) >= self.args.batch_size:
                            self.flush_video_batch(batch, metas, output)
        if batch:
            self.flush_video_batch(batch, metas, output)
        feature_dim = infer_feature_dim(output)
        array = np.zeros((len(rows), 1, feature_dim * len(modes)), dtype=np.float32)
        for row_idx, items in enumerate(output):
            vectors = [vector if vector is not None else np.zeros(feature_dim, dtype=np.float32) for vector in items]
            array[row_idx, 0] = np.concatenate(vectors)
        return array

    def flush_video_batch(
        self,
        batch: list[torch.Tensor | list[np.ndarray]],
        metas: list[tuple[int, int]],
        output: list[list[np.ndarray | None]],
    ) -> None:
        batch_size = len(batch)
        started = time.perf_counter()
        x = self.make_video_batch(batch)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            y = self.model(pixel_values=x).last_hidden_state.mean(dim=1)
        y = torch.nn.functional.normalize(y.float(), dim=1).cpu().numpy().astype(np.float32)
        self.batch_count += 1
        if self.args.log_batches:
            print(
                f"flush={self.batch_count} kind=video batch={batch_size} sec={time.perf_counter() - started:.3f} "
                f"device={self.device}",
                flush=True,
            )
        for vector, (event_index, mode_index) in zip(y, metas):
            output[event_index][mode_index] = vector
        batch.clear()
        metas.clear()


def crop_image(
    image: np.ndarray,
    record: dict[str, Any] | None,
    row: dict[str, str],
    expand: float,
    mode: str,
) -> np.ndarray:
    if mode == "full":
        return image
    attacker = row["fighter"]
    defender = "blue" if attacker == "red" else "red"
    if mode == "union":
        return crop_union(image, record, expand)
    if mode == "attacker":
        return crop_role(image, record, attacker, expand)
    if mode in {"defender", "opponent"}:
        return crop_role(image, record, defender, expand)
    if mode == "attacker_defender":
        return pair_crop(crop_role(image, record, attacker, expand), crop_role(image, record, defender, expand))
    if mode == "defender_head_body":
        return crop_defender_target_zone(image, record, defender, expand)
    if mode == "glove_target":
        return crop_glove_target(image, record, row, attacker, defender, expand)
    raise ValueError(f"unknown crop mode {mode}")


def crop_union(image: np.ndarray, record: dict[str, Any] | None, expand: float) -> np.ndarray:
    if not record:
        return image
    boxes = []
    for role in ["red", "blue"]:
        bbox = record.get("fighters", {}).get(role, {}).get("bbox")
        if bbox and len(bbox) == 4:
            boxes.append([float(value) for value in bbox])
    return crop_box(image, union_box(boxes), expand) if boxes else image


def crop_role(image: np.ndarray, record: dict[str, Any] | None, role: str, expand: float) -> np.ndarray:
    if not record:
        return image
    bbox = record.get("fighters", {}).get(role, {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return image
    return crop_box(image, [float(value) for value in bbox], expand)


def crop_defender_target_zone(
    image: np.ndarray,
    record: dict[str, Any] | None,
    defender: str,
    expand: float,
) -> np.ndarray:
    if not record:
        return image
    fighter = record.get("fighters", {}).get(defender, {})
    bbox = fighter.get("bbox")
    if not bbox or len(bbox) != 4:
        return image
    x1, y1, x2, y2 = [float(value) for value in bbox]
    height = y2 - y1
    # Head, torso, and guard region. Keeps lower body out of target crops.
    return crop_box(image, [x1, y1, x2, y1 + height * 0.68], expand)


def crop_glove_target(
    image: np.ndarray,
    record: dict[str, Any] | None,
    row: dict[str, str],
    attacker: str,
    defender: str,
    expand: float,
) -> np.ndarray:
    if not record:
        return image
    attacker_item = record.get("fighters", {}).get(attacker, {})
    defender_item = record.get("fighters", {}).get(defender, {})
    wrist = hand_wrist(attacker_item, row["hand"])
    target = target_point(defender_item, row.get("target", "head"))
    boxes = []
    for point in [wrist, target]:
        if point is not None:
            x, y = point
            boxes.append([x - 24, y - 24, x + 24, y + 24])
    defender_bbox = defender_item.get("bbox")
    if defender_bbox and len(defender_bbox) == 4:
        boxes.append([float(value) for value in defender_bbox])
    return crop_box(image, union_box(boxes), expand) if boxes else crop_union(image, record, expand)


def hand_wrist(fighter: dict[str, Any], hand: str) -> tuple[float, float] | None:
    index = 9 if hand == "left" else 10
    return keypoint_xy(fighter, index)


def target_point(fighter: dict[str, Any], target: str) -> tuple[float, float] | None:
    if target == "head":
        points = [keypoint_xy(fighter, index) for index in [0, 1, 2, 3, 4]]
        points = [point for point in points if point is not None]
        if points:
            return tuple(np.mean(np.asarray(points, dtype=np.float32), axis=0).tolist())  # type: ignore[return-value]
    bbox = fighter.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if target == "head":
        return ((x1 + x2) * 0.5, y1 + (y2 - y1) * 0.15)
    return ((x1 + x2) * 0.5, y1 + (y2 - y1) * 0.45)


def keypoint_xy(fighter: dict[str, Any], index: int) -> tuple[float, float] | None:
    keypoints = fighter.get("keypoints", [])
    if index >= len(keypoints):
        return None
    item = keypoints[index]
    if len(item) < 3 or float(item[2]) <= 0:
        return None
    return float(item[0]), float(item[1])


def crop_box(image: np.ndarray, box: list[float], expand: float) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= bw * expand
    x2 += bw * expand
    y1 -= bh * expand
    y2 += bh * expand
    xi1 = max(0, min(width - 1, int(round(x1))))
    xi2 = max(0, min(width, int(round(x2))))
    yi1 = max(0, min(height - 1, int(round(y1))))
    yi2 = max(0, min(height, int(round(y2))))
    if xi2 - xi1 < 16 or yi2 - yi1 < 16:
        return image
    return image[yi1:yi2, xi1:xi2]


def union_box(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def pair_crop(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0], 16)
    left_width = max(16, round(left.shape[1] * height / max(1, left.shape[0])))
    right_width = max(16, round(right.shape[1] * height / max(1, right.shape[0])))
    left_resized = cv2.resize(left, (left_width, height), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right, (right_width, height), interpolation=cv2.INTER_AREA)
    return np.concatenate([left_resized, right_resized], axis=1)


def to_tensor(image: np.ndarray, size: int, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    arr = image.astype(np.float32) / 255.0
    arr = (arr - mean[None, None, :]) / std[None, None, :]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr.astype(np.float32))


def as_uint8_hwc(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC BGR image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def load_track_records(path: Path, frames: set[int]) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = int(record["frame"])
            if frame in frames:
                records[frame] = record
    return records


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    return cap


def read_frame(cap: cv2.VideoCapture, frame: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    if not ok or image is None:
        raise RuntimeError(f"failed to read frame {frame}")
    return image


def iter_wanted_frames(path: Path, wanted: set[int], max_sequential_gap: int):
    if not wanted:
        return
    runs = wanted_runs(sorted(wanted), max_sequential_gap)
    cap = open_video(path)
    try:
        for first, last, run_wanted in runs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            frame = first
            while frame <= last:
                ok, image = cap.read()
                if not ok or image is None:
                    raise RuntimeError(f"failed to read frame {frame} from {path}")
                if frame in run_wanted:
                    yield frame, image
                frame += 1
    finally:
        cap.release()


def wanted_runs(wanted: list[int], max_sequential_gap: int) -> list[tuple[int, int, set[int]]]:
    if not wanted:
        return []
    runs = []
    first = wanted[0]
    prev = wanted[0]
    run = {wanted[0]}
    for frame in wanted[1:]:
        if frame - prev > max_sequential_gap:
            runs.append((first, prev, run))
            first = frame
            run = set()
        run.add(frame)
        prev = frame
    runs.append((first, prev, run))
    return runs


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["feature_index", "id", "video_key", "frame", "fighter", "hand", "target", "punch_type", "effectiveness"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "feature_index": index,
                    "id": row.get("id", ""),
                    "video_key": row["video_key"],
                    "frame": row["frame"],
                    "fighter": row["fighter"],
                    "hand": row["hand"],
                    "target": row["target"],
                    "punch_type": row.get("punch_type", ""),
                    "effectiveness": row.get("effectiveness", ""),
                }
            )


def infer_feature_dim(items: list[list[np.ndarray | None]]) -> int:
    for row in items:
        for value in row:
            if value is not None:
                return int(value.shape[0])
    raise RuntimeError("no features extracted")


def processor_size(processor: Any, fallback: int) -> int:
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        return int(size.get("height") or size.get("shortest_edge") or fallback)
    if isinstance(size, int):
        return size
    return fallback


def centered_offsets(clip_len: int, stride: int) -> list[int]:
    center = clip_len // 2
    return [(index - center) * stride for index in range(clip_len)]


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_key"]].append(row)
    return dict(grouped)


def progress(items, desc: str, quiet: bool):
    return tqdm(list(items), desc=desc, leave=False, disable=quiet, mininterval=5.0)


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def clamp_frame(frame: int, frame_count: int) -> int:
    return max(0, min(frame_count - 1, int(frame)))


if __name__ == "__main__":
    raise SystemExit(main())
