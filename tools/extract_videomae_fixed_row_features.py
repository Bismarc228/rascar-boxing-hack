#!/usr/bin/env python3
"""Extract fixed-row VideoMAE features with GPU-1 and NVIDIA decode."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, VideoMAEModel

from rascar_boxing.io import as_int, read_csv_rows


MODEL_DIR = "MCG-NJU__videomae-base-finetuned-kinetics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--video-split", choices=["train", "test"], default="train")
    parser.add_argument("--model-root", type=Path, default=Path("data/processed/model_zoo/hf"))
    parser.add_argument("--crop-modes", default="attacker_defender,glove_target")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--videomae-pooling", choices=["mean", "temporal"], default="temporal")
    parser.add_argument("--crop-expand", type=float, default=0.16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--video-decoder", choices=["ffmpeg_cuda", "opencv"], default="ffmpeg_cuda")
    parser.add_argument("--preprocess-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--log-batches", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv2.setNumThreads(max(0, int(args.cpu_threads)))
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    if args.max_rows:
        rows = rows[: args.max_rows]
    videos = read_csv_rows(args.data_root / args.video_split / "videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    missing = sorted({row["video_key"] for row in rows} - set(video_by_key))
    if missing:
        raise ValueError(f"Prediction rows are not {args.video_split} videos: {missing}")

    extractor = VideoMAEExtractor(args)
    features = extractor.extract(rows, video_by_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features.astype(np.float32),
        backbone=np.asarray(["videomae_k400"]),
        crop_modes=np.asarray(parse_list(args.crop_modes)),
        clip_len=np.asarray([args.clip_len], dtype=np.int16),
        frame_stride=np.asarray([args.frame_stride], dtype=np.int16),
        videomae_pooling=np.asarray([args.videomae_pooling]),
        video_split=np.asarray([args.video_split]),
        video_decoder=np.asarray([args.video_decoder]),
        preprocess_device=np.asarray([args.preprocess_device]),
    )
    write_index(args.index_output, rows)
    print(f"wrote_features={args.output} shape={features.shape}", flush=True)
    print(f"wrote_index={args.index_output} rows={len(rows)}", flush=True)
    return 0


class VideoMAEExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.enforce_gpu_visibility()
        model_path = args.model_root / MODEL_DIR
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.size = processor_size(self.processor, args.image_size)
        self.mean = np.asarray(getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(getattr(self.processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        if args.preprocess_device == "cuda" and self.device.type != "cuda":
            raise ValueError("--preprocess-device cuda requires a CUDA model device")
        self.mean_tensor = torch.as_tensor(self.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.as_tensor(self.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.model = VideoMAEModel.from_pretrained(model_path, local_files_only=True).eval().to(self.device)
        self.batch_count = 0
        if not args.quiet:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
            print(
                f"extractor_device={self.device} device_name={device_name} "
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
                f"video_decoder={args.video_decoder} preprocess_device={args.preprocess_device} "
                f"model_path={model_path}",
                flush=True,
            )

    def enforce_gpu_visibility(self) -> None:
        if self.device.type != "cuda":
            return
        visible = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
        if visible != ["1"]:
            raise RuntimeError(
                "VideoMAE extraction must run with CUDA_VISIBLE_DEVICES=1 "
                f"to avoid physical GPU 0; got {os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if self.device.index not in (None, 0):
            raise RuntimeError("Use --device cuda with CUDA_VISIBLE_DEVICES=1")
        self.device = torch.device("cuda:0")

    def extract(self, rows: list[dict[str, str]], video_by_key: dict[str, dict[str, str]]) -> np.ndarray:
        modes = parse_list(self.args.crop_modes)
        offsets = centered_offsets(self.args.clip_len, self.args.frame_stride)
        output: list[list[np.ndarray | None]] = [[None] * len(modes) for _ in rows]
        batch: list[torch.Tensor | list[np.ndarray]] = []
        metas: list[tuple[int, int]] = []
        grouped = group_rows(rows)
        row_index = {id(row): index for index, row in enumerate(rows)}
        for key, group in progress(grouped.items(), "videomae-features", self.args.quiet):
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
            frame_iter = self.frame_iterator(video_path, int(video["width"]), int(video["height"]), set(jobs_by_frame))
            for frame, image in frame_iter:
                record = records.get(frame)
                for clip_key, offset_index, row, mode in jobs_by_frame.get(frame, []):
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
        return video_feature_array(output, len(rows), len(modes))

    def frame_iterator(
        self,
        video_path: Path,
        width: int,
        height: int,
        wanted: set[int],
    ) -> Iterator[tuple[int, np.ndarray]]:
        if self.args.video_decoder == "ffmpeg_cuda":
            return iter_wanted_frames_ffmpeg_cuda(video_path, width, height, wanted)
        return iter_wanted_frames_opencv(video_path, wanted)

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor | np.ndarray:
        if self.args.preprocess_device == "cuda":
            return as_uint8_hwc(crop)
        return to_tensor(crop, self.size, self.mean, self.std)

    def preprocess_clip(self, slots: list[torch.Tensor | np.ndarray | None]) -> torch.Tensor | list[np.ndarray]:
        items = [item for item in slots if item is not None]
        if len(items) != len(slots):
            raise RuntimeError("incomplete VideoMAE clip")
        if self.args.preprocess_device == "cuda":
            raw = [item for item in items if isinstance(item, np.ndarray)]
            if len(raw) != len(items):
                raise RuntimeError("mixed CPU/CUDA preprocess items in video clip")
            return raw
        tensors = [item for item in items if isinstance(item, torch.Tensor)]
        if len(tensors) != len(items):
            raise RuntimeError("mixed CPU/CUDA preprocess items in video clip")
        return torch.stack(tensors)

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
        tensors = []
        for crop in crops:
            tensor = torch.from_numpy(crop).to(self.device, non_blocking=True)
            tensor = tensor[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(
                tensor.float(),
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            tensors.append(tensor)
        x = torch.cat(tensors, dim=0).div_(255.0)
        return (x - self.mean_tensor) / self.std_tensor

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
            hidden = self.model(pixel_values=x).last_hidden_state
            if self.args.videomae_pooling == "mean":
                y = torch.nn.functional.normalize(hidden.mean(dim=1).float(), dim=1)
            else:
                y = self.temporal_pool_videomae_tokens(hidden, int(x.shape[1]))
                y = torch.nn.functional.normalize(y.float(), dim=2)
        y_np = y.cpu().numpy().astype(np.float32)
        self.batch_count += 1
        if self.args.log_batches:
            print(
                f"flush={self.batch_count} batch={batch_size} sec={time.perf_counter() - started:.3f} "
                f"device={self.device}",
                flush=True,
            )
        for vector, (event_index, mode_index) in zip(y_np, metas):
            output[event_index][mode_index] = vector
        batch.clear()
        metas.clear()

    def temporal_pool_videomae_tokens(self, hidden: torch.Tensor, clip_len: int) -> torch.Tensor:
        config = self.model.config
        tubelet_size = int(getattr(config, "tubelet_size", 2))
        temporal_bins = max(1, clip_len // max(1, tubelet_size))
        token_count = int(hidden.shape[1])
        if token_count % temporal_bins != 0:
            image_size = int(getattr(config, "image_size", self.size))
            patch_size = int(getattr(config, "patch_size", 16))
            spatial_tokens = max(1, (image_size // patch_size) ** 2)
            if token_count % spatial_tokens != 0:
                raise RuntimeError(
                    f"cannot infer VideoMAE temporal bins from tokens={token_count}, "
                    f"clip_len={clip_len}, tubelet_size={tubelet_size}, spatial_tokens={spatial_tokens}"
                )
            temporal_bins = token_count // spatial_tokens
        spatial_tokens = token_count // temporal_bins
        return hidden.reshape(hidden.shape[0], temporal_bins, spatial_tokens, hidden.shape[2]).mean(dim=2)


def iter_wanted_frames_ffmpeg_cuda(
    video_path: Path,
    width: int,
    height: int,
    wanted: set[int],
) -> Iterator[tuple[int, np.ndarray]]:
    if not wanted:
        return
    max_frame = max(wanted)
    frame_size = width * height * 3
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
        "bgr24",
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
            if frame in wanted:
                image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                yield frame, np.ascontiguousarray(image)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.terminate()
        try:
            _stderr = proc.stderr.read() if proc.stderr is not None else b""
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def iter_wanted_frames_opencv(video_path: Path, wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    if not wanted:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    try:
        for frame in sorted(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok or image is None:
                raise RuntimeError(f"failed to read frame {frame} from {video_path}")
            yield frame, image
    finally:
        cap.release()


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
    return np.array(image, copy=True, order="C")


def video_feature_array(
    output: list[list[np.ndarray | None]],
    n_rows: int,
    n_modes: int,
) -> np.ndarray:
    feature_shape = infer_feature_shape(output)
    if len(feature_shape) == 1:
        feature_dim = feature_shape[0]
        array = np.zeros((n_rows, 1, feature_dim * n_modes), dtype=np.float32)
        for row_idx, items in enumerate(output):
            vectors = [vector if vector is not None else np.zeros(feature_dim, dtype=np.float32) for vector in items]
            array[row_idx, 0] = np.concatenate(vectors)
        return array
    if len(feature_shape) == 2:
        temporal_bins, feature_dim = feature_shape
        array = np.zeros((n_rows, temporal_bins, feature_dim * n_modes), dtype=np.float32)
        for row_idx, items in enumerate(output):
            vectors = [
                vector if vector is not None else np.zeros((temporal_bins, feature_dim), dtype=np.float32)
                for vector in items
            ]
            array[row_idx] = np.concatenate(vectors, axis=1)
        return array
    raise RuntimeError(f"unsupported VideoMAE feature shape {feature_shape}")


def infer_feature_shape(items: list[list[np.ndarray | None]]) -> tuple[int, ...]:
    for row in items:
        for value in row:
            if value is not None:
                return tuple(int(dim) for dim in value.shape)
    raise RuntimeError("no features extracted")


def load_track_records(path: Path, frames: set[int]) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = int(record["frame"])
            if frame in frames:
                records[frame] = record
    return records


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


def clamp_frame(frame: int, frame_count: int) -> int:
    return max(0, min(frame_count - 1, int(frame)))


if __name__ == "__main__":
    raise SystemExit(main())
