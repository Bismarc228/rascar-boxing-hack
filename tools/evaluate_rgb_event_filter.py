#!/usr/bin/env python3
"""Evaluate a frozen RGB crop encoder as a clear-event filter witness."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from tools.evaluate_pose_selection_variants import score_summary, video_wins


DEFAULT_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
DEFAULT_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--crop-expand", type=float, default=0.16)
    parser.add_argument(
        "--crop-mode",
        choices=["union", "full", "attacker", "opponent", "attacker_opponent"],
        default="union",
    )
    parser.add_argument("--frame-offsets", default="0")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    pred_rows = [row for row in read_csv_rows(args.predictions) if row["clear"] == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row["clear"] == "true" and row["video_key"] in set(keys)
    ]

    features = load_or_extract_features(args, pred_rows, video_by_key)
    labels = matched_labels(gt_rows, pred_rows)
    groups = [fight_group(video_by_key[row["video_key"]]) for row in pred_rows]
    probabilities = oof_probabilities(features, labels, groups)

    baseline = score_predictions(gt_rows, pred_rows)
    baseline_summary = score_summary(baseline)
    print(
        f"baseline score={baseline['macro_score']:.6f} "
        f"time={baseline_summary['time']:.6f} fp={baseline_summary['fp_penalty']:.6f} "
        f"n={len(pred_rows)} pos_rate={float(labels.mean()):.4f}"
    )
    print("score,delta,time,fp,wins,n_rows,threshold,dropped")
    results = []
    for threshold in parse_floats(args.thresholds):
        rows = [
            dict(row)
            for row, probability in zip(pred_rows, probabilities)
            if probability >= threshold
        ]
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        results.append((score["macro_score"], summary["time"], summary["fp_penalty"], rows, threshold, score))
    for macro, time_score, fp, rows, threshold, score in sorted(results, reverse=True):
        print(
            f"{macro:.6f},{macro - baseline['macro_score']:.6f},"
            f"{time_score:.6f},{fp:.6f},{video_wins(score, baseline)},"
            f"{len(rows)},{threshold},{len(pred_rows) - len(rows)}"
        )
    print_video_probability_summary(pred_rows, probabilities, labels)
    return 0


def load_or_extract_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
) -> np.ndarray:
    if args.feature_cache and args.feature_cache.exists():
        data = np.load(args.feature_cache)
        return data["features"].astype(np.float32)
    device = pick_device(args.device)
    model, mean, std = load_model(args.model_name, args.pretrained, device)
    frame_offsets = parse_ints(args.frame_offsets)
    tensors = []
    metas: list[tuple[int, int]] = []
    event_features: list[list[np.ndarray]] = [[] for _ in pred_rows]
    row_to_index = {id(row): index for index, row in enumerate(pred_rows)}
    for key, rows in progress(group_rows(pred_rows).items(), "rgb-features", args.quiet):
        video = video_by_key[key]
        frame_count = int(video["frame_count"])
        wanted_frames = {
            max(0, min(frame_count - 1, int(row["frame"]) + offset))
            for row in rows
            for offset in frame_offsets
        }
        records = load_track_records(args.tracks_dir / f"{key}.jsonl", wanted_frames)
        cap = cv2.VideoCapture(str(args.data_root / video["video_path"]))
        if not cap.isOpened():
            raise FileNotFoundError(args.data_root / video["video_path"])
        try:
            for row in rows:
                event_index = row_to_index[id(row)]
                center = int(row["frame"])
                for offset in frame_offsets:
                    frame = max(0, min(frame_count - 1, center + offset))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = cap.read()
                    if not ok or image is None:
                        tensor = torch.zeros(3, args.image_size, args.image_size)
                    else:
                        tensor = image_to_tensor(
                            image,
                            records.get(frame),
                            row,
                            args.image_size,
                            args.crop_expand,
                            mean,
                            std,
                            args.crop_mode,
                        )
                    tensors.append(tensor)
                    metas.append((event_index, offset))
                    if len(tensors) >= args.batch_size:
                        flush_batch(model, device, tensors, metas, event_features)
        finally:
            cap.release()
    if tensors:
        flush_batch(model, device, tensors, metas, event_features)
    output = np.stack([aggregate_event_features(items) for items in event_features]).astype(np.float32)
    if args.feature_cache:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feature_cache, features=output)
    return output


def pick_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_model(
    model_name: str,
    pretrained: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    import timm

    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    except RuntimeError as exc:
        if "fc_norm" not in str(exc) or "norm." not in str(exc):
            raise
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="token")
    model.eval().to(device)
    cfg = getattr(model, "default_cfg", {}) or {}
    mean = np.asarray(cfg.get("mean", DEFAULT_MEAN[:, 0, 0]), dtype=np.float32)[:, None, None]
    std = np.asarray(cfg.get("std", DEFAULT_STD[:, 0, 0]), dtype=np.float32)[:, None, None]
    return model, mean, std


def image_to_tensor(
    image: np.ndarray,
    record: dict[str, Any] | None,
    row: dict[str, str],
    image_size: int,
    crop_expand: float,
    mean: np.ndarray,
    std: np.ndarray,
    crop_mode: str,
) -> torch.Tensor:
    crop = crop_image(image, record, row, crop_expand, crop_mode)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = crop.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - mean) / std
    return torch.from_numpy(arr.astype(np.float32))


def crop_bounds(
    image_shape: tuple[int, ...],
    record: dict[str, Any] | None,
    row: dict[str, str],
    expand: float,
    mode: str,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    if mode == "full":
        return 0, height, 0, width
    if mode == "union":
        return crop_union_bounds(height, width, record, expand)
    attacker = row["fighter"]
    opponent = "blue" if attacker == "red" else "red"
    if mode == "attacker":
        return crop_role_bounds(height, width, record, attacker, expand)
    if mode == "opponent":
        return crop_role_bounds(height, width, record, opponent, expand)
    if mode == "attacker_opponent":
        return 0, height, 0, width
    raise ValueError(f"Unknown crop_mode: {mode}")


def crop_union_bounds(
    height: int,
    width: int,
    record: dict[str, Any] | None,
    expand: float,
) -> tuple[int, int, int, int]:
    if not record:
        return 0, height, 0, width
    boxes = []
    for role in ["red", "blue"]:
        fighter = record.get("fighters", {}).get(role, {})
        bbox = fighter.get("bbox")
        if bbox and len(bbox) == 4:
            boxes.append([float(value) for value in bbox])
    if not boxes:
        return 0, height, 0, width
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return expanded_bounds(height, width, x1, y1, x2, y2, expand)


def crop_role_bounds(
    height: int,
    width: int,
    record: dict[str, Any] | None,
    role: str,
    expand: float,
) -> tuple[int, int, int, int]:
    if not record:
        return 0, height, 0, width
    fighter = record.get("fighters", {}).get(role, {})
    bbox = fighter.get("bbox")
    if not bbox or len(bbox) != 4:
        return 0, height, 0, width
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return expanded_bounds(height, width, x1, y1, x2, y2, expand)


def expanded_bounds(
    height: int,
    width: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    expand: float,
) -> tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    x1 -= bw * expand
    x2 += bw * expand
    y1 -= bh * expand
    y2 += bh * expand
    xi1 = max(0, min(width - 1, int(round(x1))))
    xi2 = max(0, min(width, int(round(x2))))
    yi1 = max(0, min(height - 1, int(round(y1))))
    yi2 = max(0, min(height, int(round(y2))))
    if xi2 - xi1 < 16 or yi2 - yi1 < 16:
        return 0, height, 0, width
    return yi1, yi2, xi1, xi2


def crop_image(
    image: np.ndarray,
    record: dict[str, Any] | None,
    row: dict[str, str],
    expand: float,
    mode: str,
) -> np.ndarray:
    if mode == "full":
        return image
    if mode == "union":
        return crop_union(image, record, expand)
    attacker = row["fighter"]
    opponent = "blue" if attacker == "red" else "red"
    if mode == "attacker":
        return crop_role(image, record, attacker, expand)
    if mode == "opponent":
        return crop_role(image, record, opponent, expand)
    if mode == "attacker_opponent":
        return pair_crop(
            crop_role(image, record, attacker, expand),
            crop_role(image, record, opponent, expand),
        )
    raise ValueError(f"Unknown crop_mode: {mode}")


def crop_union(image: np.ndarray, record: dict[str, Any] | None, expand: float) -> np.ndarray:
    if not record:
        return image
    boxes = []
    for role in ["red", "blue"]:
        fighter = record.get("fighters", {}).get(role, {})
        bbox = fighter.get("bbox")
        if bbox and len(bbox) == 4:
            boxes.append([float(value) for value in bbox])
    if not boxes:
        return image
    height, width = image.shape[:2]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    bw = x2 - x1
    bh = y2 - y1
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


def crop_role(
    image: np.ndarray,
    record: dict[str, Any] | None,
    role: str,
    expand: float,
) -> np.ndarray:
    if not record:
        return image
    fighter = record.get("fighters", {}).get(role, {})
    bbox = fighter.get("bbox")
    if not bbox or len(bbox) != 4:
        return image
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    bw = x2 - x1
    bh = y2 - y1
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


def pair_crop(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0], 16)
    left_width = max(16, round(left.shape[1] * height / max(1, left.shape[0])))
    right_width = max(16, round(right.shape[1] * height / max(1, right.shape[0])))
    left_resized = cv2.resize(left, (left_width, height), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right, (right_width, height), interpolation=cv2.INTER_AREA)
    return np.concatenate([left_resized, right_resized], axis=1)


def run_batch(
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
) -> list[np.ndarray]:
    x = torch.stack(tensors).to(device, non_blocking=True)
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = torch.nn.functional.normalize(y.float(), dim=1).cpu().numpy()
    tensors.clear()
    return [row.astype(np.float32) for row in y]


def flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    tensors: list[torch.Tensor],
    metas: list[tuple[int, int]],
    event_features: list[list[np.ndarray]],
) -> None:
    vectors = run_batch(model, device, tensors)
    for vector, (event_index, _offset) in zip(vectors, metas):
        event_features[event_index].append(vector)
    metas.clear()


def aggregate_event_features(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        return np.zeros(2048 * 2, dtype=np.float32)
    values = np.stack(items).astype(np.float32)
    mean = values.mean(axis=0)
    if len(items) > 1:
        std = values.std(axis=0)
        return np.concatenate([mean, std]).astype(np.float32)
    return np.concatenate([mean, np.zeros_like(mean)]).astype(np.float32)


def load_track_records(path: Path, frames: set[int]) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame"])
            if frame in frames:
                records[frame] = record
    return records


def matched_labels(gt_rows: list[dict[str, str]], pred_rows: list[dict[str, str]]) -> np.ndarray:
    labels = np.zeros(len(pred_rows), dtype=np.int8)
    gt_by_key = group_rows(gt_rows)
    pred_by_key = group_rows(pred_rows)
    offset = 0
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            labels[offset + match.pred_index] = 1
        offset += len(rows)
    return labels


def oof_probabilities(features: np.ndarray, labels: np.ndarray, groups: list[str]) -> np.ndarray:
    output = np.zeros(len(labels), dtype=np.float32)
    unique_groups = sorted(set(groups))
    groups_arr = np.asarray(groups)
    for group in unique_groups:
        train = groups_arr != group
        valid = groups_arr == group
        if len(set(labels[train].tolist())) < 2:
            output[valid] = float(labels[train].mean()) if train.any() else 0.5
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.2),
        )
        model.fit(features[train], labels[train])
        output[valid] = model.predict_proba(features[valid])[:, 1].astype(np.float32)
    return output


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_key"]].append(row)
    return dict(grouped)


def fight_group(video: dict[str, str]) -> str:
    return f"{video['data_root']}::{video['fight_index']}"


def print_video_probability_summary(
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> None:
    print("video_key,n,pos_rate,mean_p,pos_mean_p,neg_mean_p")
    start = 0
    for key, items in group_rows(rows).items():
        stop = start + len(items)
        y = labels[start:stop]
        p = probabilities[start:stop]
        pos_p = p[y == 1]
        neg_p = p[y == 0]
        print(
            f"{key},{len(items)},{float(y.mean()):.4f},{float(p.mean()):.4f},"
            f"{float(pos_p.mean()) if len(pos_p) else 0.0:.4f},"
            f"{float(neg_p.mean()) if len(neg_p) else 0.0:.4f}"
        )
        start = stop


def progress(items, desc: str, quiet: bool):
    return tqdm(list(items), desc=desc, leave=False, disable=quiet, mininterval=5.0)


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
