#!/usr/bin/env python3
"""Evaluate per-video appearance clustering for fighter-label correction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from sklearn.cluster import KMeans

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import match_events, score_predictions
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
from tools.evaluate_pose_selection_variants import score_summary, video_wins


FIGHTERS = ("red", "blue")


@dataclass
class TrackAppearance:
    track_id: int
    descriptor: np.ndarray
    n_obs: int
    role_counts: Counter[str]


@dataclass
class VideoCalibration:
    track_cluster: dict[int, int]
    role_mapping: dict[int, str]
    oracle_mapping: dict[int, str]
    purity: float
    n_tracks: int
    n_obs: int
    fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--nms-frames", type=int, default=8)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--count-mode", default="threshold")
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    parser.add_argument("--context-feature", default="same_count")
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--sample-stride", type=int, default=24)
    parser.add_argument("--candidate-neighbor-frames", default="0,3,6")
    parser.add_argument("--min-track-obs", type=int, default=4)
    parser.add_argument("--purity-threshold", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

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
    calibration_by_key: dict[str, VideoCalibration] = {}
    for key in ready_keys:
        tracks_path = args.tracks_dir / f"{key}.jsonl"
        candidates = score_pose_tracks(tracks_path, PoseHeuristicConfig(min_score=0.0))
        candidates = apply_temporal_context(candidates, config)
        count = estimate_count(config, video_by_key[key], train_videos, train_gt, [])
        selected = select_candidates(candidates, config, count)
        selected_by_key[key] = selected
        sample_frames = build_sample_frames(video_by_key[key], selected, args.sample_stride, neighbor_offsets)
        track_lookup, appearances = extract_track_appearances(
            args.data_root,
            video_by_key[key],
            tracks_path,
            sample_frames,
            args.min_track_obs,
        )
        track_lookup_by_key[key] = track_lookup
        baseline_rows = build_rows_for_key(video_by_key[key], selected, attr_priors, {})
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


def complete_track_keys(tracks_dir: Path, video_by_key: dict[str, dict[str, str]]) -> list[str]:
    keys = []
    for path in sorted(tracks_dir.glob("*.jsonl")):
        video = video_by_key.get(path.stem)
        if video is None:
            continue
        with path.open("r", encoding="utf-8") as fh:
            line_count = sum(1 for _ in fh)
        if line_count == int(video["frame_count"]):
            keys.append(path.stem)
    return keys


def build_sample_frames(
    video: dict[str, str],
    selected: list[PunchCandidate],
    stride: int,
    neighbor_offsets: list[int],
) -> set[int]:
    frame_count = int(video["frame_count"])
    frames = set(range(0, frame_count, max(1, stride)))
    for candidate in selected:
        for offset in neighbor_offsets:
            frames.add(max(0, min(frame_count - 1, candidate.frame + offset)))
            frames.add(max(0, min(frame_count - 1, candidate.frame - offset)))
    return frames


def extract_track_appearances(
    data_root: Path,
    video: dict[str, str],
    tracks_path: Path,
    sample_frames: set[int],
    min_track_obs: int,
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
    try:
        for frame in sorted(records):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok or image is None:
                continue
            for role, fighter in records[frame].get("fighters", {}).items():
                if role not in FIGHTERS or "track_id" not in fighter:
                    continue
                track_id = int(fighter["track_id"])
                descriptor = describe_fighter_crop(image, fighter)
                if descriptor is None:
                    continue
                descriptors[track_id].append(descriptor)
                role_counts[track_id][role] += 1
    finally:
        cap.release()

    appearances = []
    for track_id, values in descriptors.items():
        if len(values) < min_track_obs:
            continue
        appearances.append(
            TrackAppearance(
                track_id=track_id,
                descriptor=np.median(np.stack(values), axis=0).astype(np.float32),
                n_obs=len(values),
                role_counts=role_counts[track_id],
            )
        )
    return track_lookup, appearances


def describe_fighter_crop(image: np.ndarray, fighter: dict[str, Any]) -> np.ndarray | None:
    bbox = fighter.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height, y2))
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    crop = image[y1:y2, x1:x2]
    h = crop.shape[0]
    upper = crop[int(0.18 * h) : int(0.58 * h)]
    lower = crop[int(0.48 * h) : int(0.88 * h)]
    return np.concatenate([region_descriptor(upper), region_descriptor(lower)]).astype(np.float32)


def region_descriptor(region: np.ndarray) -> np.ndarray:
    if region.size == 0:
        return np.zeros(29, dtype=np.float32)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    mask = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 35)
    pixels = region[mask]
    hsv_pixels = hsv[mask]
    lab_pixels = lab[mask]
    if len(pixels) < 20:
        pixels = region.reshape(-1, 3)
        hsv_pixels = hsv.reshape(-1, 3)
        lab_pixels = lab.reshape(-1, 3)
    hist = np.histogram(hsv_pixels[:, 0], bins=16, range=(0, 180))[0].astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    bgr = pixels.astype(np.float32) / 255.0
    denom = np.maximum(1e-3, bgr.sum(axis=1, keepdims=True))
    chroma = bgr / denom
    hsv_mean = hsv_pixels.astype(np.float32).mean(axis=0) / np.array([180.0, 255.0, 255.0], dtype=np.float32)
    lab_mean = lab_pixels.astype(np.float32).mean(axis=0) / 255.0
    chroma_mean = chroma.mean(axis=0)
    fractions = color_fractions(hsv_pixels)
    return np.concatenate([hist, hsv_mean, lab_mean, chroma_mean, fractions]).astype(np.float32)


def color_fractions(hsv_pixels: np.ndarray) -> np.ndarray:
    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]
    red = (((h < 12) | (h > 165)) & (s > 45) & (v > 45)).mean()
    blue = ((h > 90) & (h < 132) & (s > 45) & (v > 45)).mean()
    white = ((s < 35) & (v > 145)).mean()
    black = (v < 55).mean()
    return np.asarray([red, blue, white, black], dtype=np.float32)


def matched_cluster_labels(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    selected: list[PunchCandidate],
    track_lookup: dict[tuple[int, str], int],
) -> list[tuple[int, str]]:
    labels = []
    for match in match_events(gt_rows, pred_rows):
        pred = pred_rows[match.pred_index]
        candidate = selected[match.pred_index]
        track_id = track_lookup.get((candidate.frame, candidate.fighter))
        if track_id is None:
            continue
        labels.append((track_id, gt_rows[match.gt_index]["fighter"]))
    return labels


def calibrate_video(
    appearances: list[TrackAppearance],
    labels: list[tuple[int, str]],
    purity_threshold: float,
) -> VideoCalibration:
    n_obs = sum(item.n_obs for item in appearances)
    if len(appearances) < 2:
        return VideoCalibration({}, {}, {}, 0.0, len(appearances), n_obs, True)
    x = np.stack([item.descriptor for item in appearances])
    weights = np.asarray([item.n_obs for item in appearances], dtype=np.float32)
    model = KMeans(n_clusters=2, n_init=20, random_state=42)
    cluster_ids = model.fit_predict(x, sample_weight=weights)
    track_cluster = {
        item.track_id: int(cluster_id) for item, cluster_id in zip(appearances, cluster_ids)
    }
    role_mapping, purity = role_majority_mapping(appearances, track_cluster)
    oracle_mapping = oracle_cluster_mapping(labels, track_cluster, role_mapping)
    fallback = purity < purity_threshold or len(set(track_cluster.values())) < 2
    return VideoCalibration(
        track_cluster=track_cluster,
        role_mapping=role_mapping,
        oracle_mapping=oracle_mapping,
        purity=purity,
        n_tracks=len(appearances),
        n_obs=n_obs,
        fallback=fallback,
    )


def role_majority_mapping(
    appearances: list[TrackAppearance],
    track_cluster: dict[int, int],
) -> tuple[dict[int, str], float]:
    cluster_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for item in appearances:
        cluster = track_cluster[item.track_id]
        cluster_counts[cluster].update(item.role_counts)
    mapping = {}
    purities = []
    for cluster in [0, 1]:
        counts = cluster_counts.get(cluster, Counter())
        if not counts:
            mapping[cluster] = "red" if cluster == 0 else "blue"
            purities.append(0.0)
            continue
        label, count = counts.most_common(1)[0]
        mapping[cluster] = label
        purities.append(count / max(1, sum(counts.values())))
    if mapping.get(0) == mapping.get(1):
        mapping[1] = "blue" if mapping[0] == "red" else "red"
    return mapping, float(min(purities))


def oracle_cluster_mapping(
    labels: list[tuple[int, str]],
    track_cluster: dict[int, int],
    fallback: dict[int, str],
) -> dict[int, str]:
    best_mapping = dict(fallback)
    best_score = -1
    for red_cluster, blue_cluster in itertools.permutations([0, 1], 2):
        mapping = {red_cluster: "red", blue_cluster: "blue"}
        score = 0
        for track_id, gt_fighter in labels:
            cluster = track_cluster.get(track_id)
            if cluster is not None and mapping.get(cluster) == gt_fighter:
                score += 1
        if score > best_score:
            best_score = score
            best_mapping = mapping
    return best_mapping


def build_rows(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    selected_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, Any],
    cluster_mapping_by_key: dict[str, dict[int, str]],
    track_lookup_by_key: dict[str, dict[tuple[int, str], int]],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        rows.extend(
            build_rows_for_key(
                video_by_key[key],
                selected_by_key[key],
                attr_priors,
                cluster_mapping_by_key.get(key, {}),
                track_lookup_by_key[key],
                row_id,
            )
        )
        row_id += len(selected_by_key[key])
    return rows


def track_fighter_mapping(
    calibration: VideoCalibration,
    cluster_mapping: dict[int, str],
) -> dict[int, str]:
    return {
        track_id: cluster_mapping.get(cluster, "red" if cluster == 0 else "blue")
        for track_id, cluster in calibration.track_cluster.items()
    }


def build_rows_for_key(
    video: dict[str, str],
    selected: list[PunchCandidate],
    attr_priors: dict[str, Any],
    cluster_mapping: dict[int, str],
    track_lookup: dict[tuple[int, str], int] | None = None,
    start_id: int = 1,
) -> list[dict[str, str]]:
    track_lookup = track_lookup or {}
    rows = []
    for index, candidate in enumerate(sorted(selected, key=lambda item: item.frame)):
        attrs = estimate_attrs(candidate, attr_priors)
        track_id = track_lookup.get((candidate.frame, candidate.fighter))
        fighter = cluster_mapping.get(track_id, candidate.fighter)
        rows.append(
            {
                "id": str(start_id + index),
                "video_id": video["video_id"],
                "agn_index": video["agn_index"],
                "video_key": video["video_key"],
                "frame": str(candidate.frame),
                "fighter": fighter,
                "punch_type": attrs["punch_type"],
                "hand": candidate.hand,
                "target": candidate.target,
                "effectiveness": attrs["effectiveness"],
                "clear": "true",
            }
        )
    return rows


def group_by_key(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def count_changed(left: list[dict[str, str]], right: list[dict[str, str]]) -> int:
    return sum(1 for a, b in zip(left, right) if a["fighter"] != b["fighter"])


def score_summary_with_fighter(score: dict[str, object]) -> dict[str, float]:
    summary = score_summary(score)
    summary["fighter"] = float(np.mean([item["score_fighter"] for item in score["by_video"].values()]))
    return summary


def print_root_deltas(
    scores: dict[str, dict[str, object]],
    baseline: dict[str, object],
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
) -> None:
    by_root: dict[str, list[str]] = defaultdict(list)
    for key in ready_keys:
        by_root[video_by_key[key]["data_root"]].append(key)
    print("root,variant,macro_delta,fighter_delta,n_videos")
    for root, keys in sorted(by_root.items()):
        for label, score in scores.items():
            macro_delta = np.mean(
                [
                    score["by_video"][key]["final_score"] - baseline["by_video"][key]["final_score"]
                    for key in keys
                ]
            )
            fighter_delta = np.mean(
                [
                    score["by_video"][key]["score_fighter"]
                    - baseline["by_video"][key]["score_fighter"]
                    for key in keys
                ]
            )
            print(f"{root},{label},{macro_delta:.6f},{fighter_delta:.6f},{len(keys)}")


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
