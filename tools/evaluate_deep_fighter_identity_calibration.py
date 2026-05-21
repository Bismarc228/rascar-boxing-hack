#!/usr/bin/env python3
"""Evaluate per-video deep crop embeddings for fighter-label correction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm

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
from tools.evaluate_per_video_fighter_identity_calibration import (
    FIGHTERS,
    TrackAppearance,
    build_rows,
    build_sample_frames,
    calibrate_video,
    complete_track_keys,
    count_changed,
    group_by_key,
    matched_cluster_labels,
    print_root_deltas,
    score_summary_with_fighter,
    track_fighter_mapping,
)
from tools.evaluate_pose_selection_variants import video_wins


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--nms-frames", type=int, default=10)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--count-mode", default="root_rate")
    parser.add_argument("--count-multiplier", type=float, default=0.88)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--sample-stride", type=int, default=48)
    parser.add_argument("--candidate-neighbor-frames", default="0,4,8")
    parser.add_argument("--min-track-obs", type=int, default=3)
    parser.add_argument("--purity-threshold", type=float, default=0.70)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-expand", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-keys", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if args.video_keys:
        requested = {value for value in args.video_keys.split(",") if value}
        ready_keys = [key for key in ready_keys if key in requested]
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

    device = pick_device(args.device)
    model = load_embedder(args.model_name, args.pretrained, device)
    print(
        f"ready={len(ready_keys)} model={args.model_name} pretrained={int(args.pretrained)} "
        f"device={device} stride={args.sample_stride}",
        flush=True,
    )

    ready_set = set(ready_keys)
    train_videos = [row for row in videos if row["video_key"] not in ready_set]
    gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in ready_set]
    gt_by_key = group_by_key(gt, "video_key")
    train_gt = [
        row for row in punches if row["clear"] == "true" and row["video_key"] not in ready_set
    ]
    attr_priors = fit_attr_priors(train_gt)
    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode="fighter_hand",
        cross_nms_frames=args.cross_nms_frames,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    neighbor_offsets = parse_ints(args.candidate_neighbor_frames)

    selected_by_key: dict[str, list[PunchCandidate]] = {}
    track_lookup_by_key: dict[str, dict[tuple[int, str], int]] = {}
    calibration_by_key = {}
    for key in progress(ready_keys, "deep-identity", args.quiet):
        tracks_path = args.tracks_dir / f"{key}.jsonl"
        candidates = score_pose_tracks(tracks_path, PoseHeuristicConfig(min_score=0.0))
        candidates = apply_temporal_context(candidates, config)
        count = estimate_count(config, video_by_key[key], train_videos, train_gt, [])
        selected = select_candidates(candidates, config, count)
        selected_by_key[key] = selected
        sample_frames = build_sample_frames(video_by_key[key], selected, args.sample_stride, neighbor_offsets)
        track_lookup, appearances = extract_deep_track_appearances(
            args.data_root,
            video_by_key[key],
            tracks_path,
            sample_frames,
            args.min_track_obs,
            model,
            device,
            args.batch_size,
            args.image_size,
            args.crop_expand,
        )
        track_lookup_by_key[key] = track_lookup
        baseline_rows = build_rows_for_key(video_by_key[key], selected, attr_priors)
        labels = matched_cluster_labels(gt_by_key.get(key, []), baseline_rows, selected, track_lookup)
        calibration_by_key[key] = calibrate_video(appearances, labels, args.purity_threshold)
        cal = calibration_by_key[key]
        print(
            f"video={key} tracks={cal.n_tracks} obs={cal.n_obs} "
            f"purity={cal.purity:.3f} fallback={int(cal.fallback)}",
            flush=True,
        )

    baseline_rows = build_rows(ready_keys, video_by_key, selected_by_key, attr_priors, {}, track_lookup_by_key)
    baseline_score = score_predictions(gt, baseline_rows)
    variants = {
        "baseline": baseline_rows,
        "role_map_gated": build_rows(
            ready_keys,
            video_by_key,
            selected_by_key,
            attr_priors,
            {
                key: track_fighter_mapping(calibration_by_key[key], calibration_by_key[key].role_mapping)
                if not calibration_by_key[key].fallback
                else {}
                for key in ready_keys
            },
            track_lookup_by_key,
        ),
        "role_map_all": build_rows(
            ready_keys,
            video_by_key,
            selected_by_key,
            attr_priors,
            {
                key: track_fighter_mapping(calibration_by_key[key], calibration_by_key[key].role_mapping)
                for key in ready_keys
            },
            track_lookup_by_key,
        ),
        "oracle_cluster_map": build_rows(
            ready_keys,
            video_by_key,
            selected_by_key,
            attr_priors,
            {
                key: track_fighter_mapping(calibration_by_key[key], calibration_by_key[key].oracle_mapping)
                for key in ready_keys
            },
            track_lookup_by_key,
        ),
    }
    print("variant,score,delta,fighter,time,fp,wins,n_pred,n_changed")
    scores: dict[str, dict[str, Any]] = {}
    for label, rows in variants.items():
        score = score_predictions(gt, rows)
        scores[label] = score
        summary = score_summary_with_fighter(score)
        print(
            f"{label},{score['macro_score']:.6f},"
            f"{score['macro_score'] - baseline_score['macro_score']:.6f},"
            f"{summary['fighter']:.6f},{summary['time']:.6f},"
            f"{summary['fp_penalty']:.6f},{video_wins(score, baseline_score)},"
            f"{len(rows)},{count_changed(baseline_rows, rows)}"
        )
    print_root_deltas(scores, baseline_score, ready_keys, video_by_key)
    return 0


def pick_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_embedder(model_name: str, pretrained: bool, device: torch.device) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    model.eval().to(device)
    return model


def extract_deep_track_appearances(
    data_root: Path,
    video: dict[str, str],
    tracks_path: Path,
    sample_frames: set[int],
    min_track_obs: int,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    crop_expand: float,
) -> tuple[dict[tuple[int, str], int], list[TrackAppearance]]:
    records = {}
    track_lookup: dict[tuple[int, str], int] = {}
    with tracks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame"])
            if frame in sample_frames:
                records[frame] = record
            for role, fighter in record.get("fighters", {}).items():
                if role in FIGHTERS and "track_id" in fighter:
                    track_lookup[(frame, role)] = int(fighter["track_id"])

    cap = cv2.VideoCapture(str(data_root / video["video_path"]))
    if not cap.isOpened():
        raise FileNotFoundError(data_root / video["video_path"])

    descriptors: dict[int, list[np.ndarray]] = defaultdict(list)
    role_counts: dict[int, Counter[str]] = defaultdict(Counter)
    batch: list[torch.Tensor] = []
    batch_meta: list[tuple[int, str]] = []
    try:
        for frame in sorted(records):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok or image is None:
                continue
            for role, fighter in records[frame].get("fighters", {}).items():
                if role not in FIGHTERS or "track_id" not in fighter:
                    continue
                tensor = crop_to_tensor(image, fighter, image_size, crop_expand)
                if tensor is None:
                    continue
                track_id = int(fighter["track_id"])
                batch.append(tensor)
                batch_meta.append((track_id, role))
                role_counts[track_id][role] += 1
                if len(batch) >= batch_size:
                    flush_batch(model, device, batch, batch_meta, descriptors)
        if batch:
            flush_batch(model, device, batch, batch_meta, descriptors)
    finally:
        cap.release()

    appearances = []
    for track_id, values in descriptors.items():
        if len(values) < min_track_obs:
            continue
        descriptor = normalize_np(np.mean(np.stack(values), axis=0).astype(np.float32))
        appearances.append(
            TrackAppearance(
                track_id=track_id,
                descriptor=descriptor,
                n_obs=len(values),
                role_counts=role_counts[track_id],
            )
        )
    return track_lookup, appearances


def crop_to_tensor(
    image: np.ndarray,
    fighter: dict[str, Any],
    image_size: int,
    crop_expand: float,
) -> torch.Tensor | None:
    bbox = fighter.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 12 or box_h < 24:
        return None
    x1 -= box_w * crop_expand
    x2 += box_w * crop_expand
    y1 -= box_h * crop_expand
    y2 += box_h * crop_expand
    xi1 = max(0, min(width - 1, int(round(x1))))
    xi2 = max(0, min(width, int(round(x2))))
    yi1 = max(0, min(height - 1, int(round(y1))))
    yi2 = max(0, min(height, int(round(y2))))
    if xi2 - xi1 < 12 or yi2 - yi1 < 24:
        return None
    crop = image[yi1:yi2, xi1:xi2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = crop.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.astype(np.float32))


def flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch: list[torch.Tensor],
    batch_meta: list[tuple[int, str]],
    descriptors: dict[int, list[np.ndarray]],
) -> None:
    x = torch.stack(batch).to(device, non_blocking=True)
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = torch.nn.functional.normalize(y.float(), dim=1).cpu().numpy()
    for vector, (track_id, _role) in zip(y, batch_meta):
        descriptors[track_id].append(vector.astype(np.float32))
    batch.clear()
    batch_meta.clear()


def build_rows_for_key(
    video: dict[str, str],
    selected: list[PunchCandidate],
    attr_priors: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []
    for index, candidate in enumerate(sorted(selected, key=lambda item: item.frame), start=1):
        attrs = estimate_attrs(candidate, attr_priors)
        rows.append(
            {
                "id": str(index),
                "video_id": video["video_id"],
                "agn_index": video["agn_index"],
                "video_key": video["video_key"],
                "frame": str(candidate.frame),
                "fighter": candidate.fighter,
                "punch_type": attrs["punch_type"],
                "hand": candidate.hand,
                "target": candidate.target,
                "effectiveness": attrs["effectiveness"],
                "clear": "true",
            }
        )
    return rows


def normalize_np(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        return values
    return (values / norm).astype(np.float32)


def progress(items: list[str], desc: str, quiet: bool):
    return tqdm(items, desc=desc, leave=False, disable=quiet, mininterval=5.0)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
