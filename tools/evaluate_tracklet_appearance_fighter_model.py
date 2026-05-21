#!/usr/bin/env python3
"""Evaluate fixed-row fighter flips from per-video tracklet appearance features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from tools.evaluate_exchange_state_gate import fight_group
from tools.evaluate_fighter_rival_flip import score_summary
from tools.evaluate_pose_selection_variants import video_wins


FIGHTERS = ("red", "blue")
DESC_DIM = 58


@dataclass(frozen=True)
class FrameFighter:
    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    score_red: float
    score_blue: float


@dataclass
class TrackSummary:
    track_id: int
    role_majority: str
    role_purity: float
    red_mean: float
    blue_mean: float
    n_frames: int
    first_frame: int
    last_frame: int
    descriptor: np.ndarray | None
    n_desc: int


@dataclass
class VideoAppearance:
    video_key: str
    frame_role_track: dict[tuple[int, str], int]
    frame_role_record: dict[tuple[int, str], FrameFighter]
    frame_descriptor: dict[tuple[int, str], np.ndarray]
    summaries: dict[int, TrackSummary]
    prototypes: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=0)
    parser.add_argument("--neighbor-frames", default="0,3,6,12")
    parser.add_argument("--min-track-obs", type=int, default=3)
    parser.add_argument("--prototype-purity", type=float, default=0.80)
    parser.add_argument("--crop-expand", type=float, default=0.08)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--models", default="logreg,hgb")
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90,0.95,0.98")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-model", default="")
    parser.add_argument("--write-threshold", type=float)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    if not pred_rows:
        print("No clear prediction rows found")
        return 2
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    missing = sorted(set(keys) - set(video_by_key))
    if missing:
        raise RuntimeError(f"Prediction keys are not train videos: {missing}")

    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, 0, len(pred_rows))

    rows_by_key = group_by(pred_rows, "video_key")
    neighbor_offsets = parse_ints(args.neighbor_frames)
    appearances = build_appearances(
        args.data_root,
        args.tracks_dir,
        video_by_key,
        rows_by_key,
        keys,
        args.sample_stride,
        neighbor_offsets,
        args.min_track_obs,
        args.prototype_purity,
        args.crop_expand,
        max(1, args.jobs),
        args.quiet,
    )
    for key in keys:
        app = appearances[key]
        n_desc_tracks = sum(1 for item in app.summaries.values() if item.descriptor is not None)
        print(
            f"video={key} tracks={len(app.summaries)} desc_tracks={n_desc_tracks} "
            f"frame_desc={len(app.frame_descriptor)}",
            flush=True,
        )

    x, action_mask = build_feature_matrix(pred_rows, video_by_key, appearances)
    labels = flip_labels(pred_rows, gt_rows)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    print(
        f"features rows={len(pred_rows)} dim={x.shape[1]} action={int(action_mask.sum())} "
        f"labels_pos={int(labels.sum())}",
        flush=True,
    )

    probabilities_by_model: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    for model_name in parse_list(args.models):
        probabilities = fit_oof_probabilities(model_name, x, labels, action_mask, groups)
        probabilities_by_model[model_name] = probabilities
        print_label_summary(model_name, labels, action_mask, probabilities)
        for threshold in parse_floats(args.thresholds):
            rows, changed = apply_flips(pred_rows, probabilities >= threshold)
            if changed == 0:
                continue
            score = score_predictions(gt_rows, rows)
            summary = score_summary(score)
            results.append(
                {
                    "score": score["macro_score"],
                    "delta": score["macro_score"] - baseline["macro_score"],
                    "fighter": summary["fighter"],
                    "time": summary["time"],
                    "hand": summary["hand"],
                    "target": summary["target"],
                    "fp": summary["fp"],
                    "wins": video_wins(score, baseline),
                    "changed": changed,
                    "model": model_name,
                    "threshold": threshold,
                }
            )

    print("score,delta,fighter,time,hand,target,fp,wins,n_changed,model,threshold")
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print(
            f"{item['score']:.6f},{item['delta']:.6f},{item['fighter']:.6f},"
            f"{item['time']:.6f},{item['hand']:.6f},{item['target']:.6f},"
            f"{item['fp']:.6f},{item['wins']},{item['changed']},"
            f"{item['model']},{item['threshold']}"
        )

    if args.write_oof_rows:
        if args.write_model and args.write_threshold is not None:
            model_name = args.write_model
            threshold = args.write_threshold
        elif results:
            best = max(results, key=lambda row: row["score"])
            model_name = str(best["model"])
            threshold = float(best["threshold"])
        else:
            print("No positive flip result; not writing OOF rows")
            return 0
        rows, changed = apply_flips(pred_rows, probabilities_by_model[model_name] >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} model={model_name} "
            f"threshold={threshold} changed={changed}",
            flush=True,
        )
    return 0


def build_appearances(
    data_root: Path,
    tracks_dir: Path,
    video_by_key: dict[str, dict[str, str]],
    rows_by_key: dict[str, list[dict[str, str]]],
    keys: list[str],
    sample_stride: int,
    neighbor_offsets: list[int],
    min_track_obs: int,
    prototype_purity: float,
    crop_expand: float,
    jobs: int,
    quiet: bool,
) -> dict[str, VideoAppearance]:
    output: dict[str, VideoAppearance] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                load_video_appearance,
                data_root,
                tracks_dir / f"{key}.jsonl",
                video_by_key[key],
                rows_by_key[key],
                sample_stride,
                neighbor_offsets,
                min_track_obs,
                prototype_purity,
                crop_expand,
            ): key
            for key in keys
        }
        iterator = as_completed(futures)
        if not quiet:
            iterator = tqdm(iterator, total=len(futures), desc="tracklet-appearance", mininterval=5.0)
        for future in iterator:
            key = futures[future]
            output[key] = future.result()
    return output


def load_video_appearance(
    data_root: Path,
    tracks_path: Path,
    video: dict[str, str],
    rows: list[dict[str, str]],
    sample_stride: int,
    neighbor_offsets: list[int],
    min_track_obs: int,
    prototype_purity: float,
    crop_expand: float,
) -> VideoAppearance:
    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)
    sample_frames = build_sample_frames(video, rows, sample_stride, neighbor_offsets)
    frame_role_track: dict[tuple[int, str], int] = {}
    frame_role_record: dict[tuple[int, str], FrameFighter] = {}
    sampled_fighters: dict[int, dict[str, FrameFighter]] = defaultdict(dict)
    role_counts: dict[int, Counter[str]] = defaultdict(Counter)
    score_sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    frame_bounds: dict[int, list[int]] = {}

    with tracks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame"])
            for role, fighter in record.get("fighters", {}).items():
                if role not in FIGHTERS or "track_id" not in fighter:
                    continue
                item = simplify_fighter(fighter)
                if item is None:
                    continue
                frame_role_track[(frame, role)] = item.track_id
                frame_role_record[(frame, role)] = item
                if frame in sample_frames:
                    sampled_fighters[frame][role] = item
                role_counts[item.track_id][role] += 1
                score_sums[item.track_id][0] += item.score_red
                score_sums[item.track_id][1] += item.score_blue
                score_sums[item.track_id][2] += 1.0
                if item.track_id not in frame_bounds:
                    frame_bounds[item.track_id] = [frame, frame]
                else:
                    frame_bounds[item.track_id][0] = min(frame_bounds[item.track_id][0], frame)
                    frame_bounds[item.track_id][1] = max(frame_bounds[item.track_id][1], frame)

    frame_descriptor, descriptors_by_track = extract_crop_descriptors(
        data_root / video["video_path"],
        sampled_fighters,
        crop_expand,
    )
    summaries: dict[int, TrackSummary] = {}
    for track_id, counts in role_counts.items():
        role_majority, role_count = counts.most_common(1)[0]
        red_sum, blue_sum, n_seen = score_sums[track_id]
        desc_values = descriptors_by_track.get(track_id, [])
        descriptor = None
        if len(desc_values) >= min_track_obs:
            descriptor = np.median(np.stack(desc_values), axis=0).astype(np.float32)
        first_frame, last_frame = frame_bounds[track_id]
        summaries[track_id] = TrackSummary(
            track_id=track_id,
            role_majority=role_majority,
            role_purity=role_count / max(1.0, n_seen),
            red_mean=red_sum / max(1.0, n_seen),
            blue_mean=blue_sum / max(1.0, n_seen),
            n_frames=int(n_seen),
            first_frame=first_frame,
            last_frame=last_frame,
            descriptor=descriptor,
            n_desc=len(desc_values),
        )
    prototypes = build_role_prototypes(summaries, prototype_purity)
    return VideoAppearance(
        video_key=video["video_key"],
        frame_role_track=frame_role_track,
        frame_role_record=frame_role_record,
        frame_descriptor=frame_descriptor,
        summaries=summaries,
        prototypes=prototypes,
    )


def build_sample_frames(
    video: dict[str, str],
    rows: list[dict[str, str]],
    sample_stride: int,
    neighbor_offsets: list[int],
) -> set[int]:
    frame_count = as_int(video["frame_count"], "frame_count")
    frames = set(range(0, frame_count, sample_stride)) if sample_stride > 0 else set()
    for row in rows:
        frame = as_int(row["frame"], "frame")
        for offset in neighbor_offsets:
            frames.add(max(0, min(frame_count - 1, frame + offset)))
            frames.add(max(0, min(frame_count - 1, frame - offset)))
    return frames


def simplify_fighter(fighter: dict[str, Any]) -> FrameFighter | None:
    bbox = fighter.get("bbox")
    center = fighter.get("center")
    if not bbox or len(bbox) != 4 or not center or len(center) != 2:
        return None
    return FrameFighter(
        track_id=int(fighter["track_id"]),
        bbox=tuple(float(value) for value in bbox),
        center=(float(center[0]), float(center[1])),
        score_red=float(fighter.get("score_red", 0.0)),
        score_blue=float(fighter.get("score_blue", 0.0)),
    )


def extract_crop_descriptors(
    video_path: Path,
    sampled_fighters: dict[int, dict[str, FrameFighter]],
    crop_expand: float,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[int, list[np.ndarray]]]:
    from decord import VideoReader, cpu

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    frame_descriptor: dict[tuple[int, str], np.ndarray] = {}
    descriptors_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    frames = [frame for frame in sorted(sampled_fighters) if 0 <= frame < len(reader)]
    if not frames:
        return frame_descriptor, descriptors_by_track
    batch = reader.get_batch(frames).asnumpy()
    for frame, image_rgb in zip(frames, batch):
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        for role, fighter in sampled_fighters[frame].items():
            descriptor = describe_fighter_crop(image, fighter, crop_expand)
            if descriptor is None:
                continue
            frame_descriptor[(frame, role)] = descriptor
            descriptors_by_track[fighter.track_id].append(descriptor)
    return frame_descriptor, descriptors_by_track


def describe_fighter_crop(
    image: np.ndarray,
    fighter: FrameFighter,
    crop_expand: float,
) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = fighter.bbox
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
    return np.concatenate([hist, hsv_mean, lab_mean, chroma_mean, color_fractions(hsv_pixels)]).astype(np.float32)


def color_fractions(hsv_pixels: np.ndarray) -> np.ndarray:
    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]
    red = (((h < 12) | (h > 165)) & (s > 45) & (v > 45)).mean()
    blue = ((h > 90) & (h < 132) & (s > 45) & (v > 45)).mean()
    white = ((s < 35) & (v > 145)).mean()
    black = (v < 55).mean()
    return np.asarray([red, blue, white, black], dtype=np.float32)


def build_role_prototypes(
    summaries: dict[int, TrackSummary],
    prototype_purity: float,
) -> dict[str, np.ndarray]:
    output = {fighter: np.zeros(DESC_DIM, dtype=np.float32) for fighter in FIGHTERS}
    for fighter in FIGHTERS:
        values = []
        weights = []
        for summary in summaries.values():
            if summary.descriptor is None:
                continue
            if summary.role_majority != fighter or summary.role_purity < prototype_purity:
                continue
            values.append(summary.descriptor)
            weights.append(max(1, summary.n_desc))
        if values:
            arr = np.stack(values)
            weight = np.asarray(weights, dtype=np.float32)[:, None]
            output[fighter] = (arr * weight).sum(axis=0) / np.maximum(1.0, weight.sum())
    return output


def build_feature_matrix(
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    appearances: dict[str, VideoAppearance],
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    action_mask = []
    for row in rows:
        app = appearances[row["video_key"]]
        vector, actionable = row_features(row, video_by_key[row["video_key"]], app)
        features.append(vector)
        action_mask.append(actionable)
    return np.stack(features).astype(np.float32), np.asarray(action_mask, dtype=bool)


def row_features(
    row: dict[str, str],
    video: dict[str, str],
    app: VideoAppearance,
) -> tuple[np.ndarray, bool]:
    frame = as_int(row["frame"], "frame")
    fighter = row["fighter"]
    other = "blue" if fighter == "red" else "red"
    current = app.frame_role_record.get((frame, fighter))
    opposite = app.frame_role_record.get((frame, other))
    current_summary = app.summaries.get(current.track_id) if current else None
    opposite_summary = app.summaries.get(opposite.track_id) if opposite else None
    current_desc = (
        current_summary.descriptor
        if current_summary is not None and current_summary.descriptor is not None
        else np.zeros(DESC_DIM, dtype=np.float32)
    )
    opposite_desc = (
        opposite_summary.descriptor
        if opposite_summary is not None and opposite_summary.descriptor is not None
        else np.zeros(DESC_DIM, dtype=np.float32)
    )
    frame_desc = app.frame_descriptor.get((frame, fighter), current_desc)
    opposite_frame_desc = app.frame_descriptor.get((frame, other), opposite_desc)
    red_proto = app.prototypes["red"]
    blue_proto = app.prototypes["blue"]
    current_proto = app.prototypes[fighter]
    other_proto = app.prototypes[other]

    base = [
        1.0 if fighter == "red" else 0.0,
        1.0 if row["hand"] == "left" else 0.0,
        1.0 if row["target"] == "head" else 0.0,
        float(frame / max(1.0, float(video["frame_count"]))),
    ]
    for root in ["бокс", "Турнир Бокс", "Турнир Бокс 2"]:
        base.append(float(video["data_root"] == root))
    base.extend(summary_features(current_summary, fighter))
    base.extend(summary_features(opposite_summary, other))
    base.extend(frame_features(current, video, fighter))
    base.extend(frame_features(opposite, video, other))
    base.extend(
        [
            float(current is not None),
            float(opposite is not None),
            l2(current_desc, current_proto),
            l2(current_desc, other_proto),
            l2(current_desc, current_proto) - l2(current_desc, other_proto),
            l2(opposite_desc, current_proto),
            l2(opposite_desc, other_proto),
            l2(opposite_desc, other_proto) - l2(opposite_desc, current_proto),
            l2(frame_desc, current_desc),
            l2(frame_desc, current_proto) - l2(frame_desc, other_proto),
            l2(opposite_frame_desc, opposite_desc),
            l2(current_desc, opposite_desc),
        ]
    )
    descriptor_block = np.concatenate(
        [
            current_desc,
            opposite_desc,
            current_desc - opposite_desc,
            current_desc - current_proto,
            current_desc - other_proto,
        ]
    )
    actionable = current_summary is not None and current_summary.descriptor is not None
    return np.concatenate([np.asarray(base, dtype=np.float32), descriptor_block.astype(np.float32)]), actionable


def summary_features(summary: TrackSummary | None, row_role: str) -> list[float]:
    if summary is None:
        return [0.0] * 11
    sign = 1.0 if row_role == "red" else -1.0
    margin = summary.red_mean - summary.blue_mean
    return [
        1.0,
        float(summary.role_majority == row_role),
        float(summary.role_majority == "red"),
        summary.role_purity,
        sign * margin,
        abs(margin),
        summary.red_mean,
        summary.blue_mean,
        np.log1p(summary.n_frames),
        np.log1p(summary.n_desc),
        float(summary.n_desc > 0),
    ]


def frame_features(record: FrameFighter | None, video: dict[str, str], row_role: str) -> list[float]:
    if record is None:
        return [0.0] * 13
    width = max(1.0, float(video["width"]))
    height = max(1.0, float(video["height"]))
    x1, y1, x2, y2 = record.bbox
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    sign = 1.0 if row_role == "red" else -1.0
    margin = record.score_red - record.score_blue
    return [
        1.0,
        record.center[0] / width,
        record.center[1] / height,
        box_w / width,
        box_h / height,
        (box_w * box_h) / max(1.0, width * height),
        box_w / max(1.0, box_h),
        sign * margin,
        abs(margin),
        record.score_red,
        record.score_blue,
        float(record.score_red >= record.score_blue),
        float(record.track_id),
    ]


def l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)))


def flip_labels(pred_rows: list[dict[str, str]], gt_rows: list[dict[str, str]]) -> np.ndarray:
    labels = np.zeros(len(pred_rows), dtype=np.int8)
    gt_by_key = group_by(gt_rows, "video_key")
    indexed_by_key: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        indexed_by_key[row["video_key"]].append((index, row))
    for key, indexed_rows in indexed_by_key.items():
        rows = [row for _, row in indexed_rows]
        for match in match_events(gt_by_key.get(key, []), rows):
            pred = rows[match.pred_index]
            gt = gt_by_key[key][match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) / FPS / 0.5 >= 1.0:
                continue
            if pred["fighter"] != gt["fighter"]:
                labels[indexed_rows[match.pred_index][0]] = 1
    return labels


def fit_oof_probabilities(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    action_mask: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    output = np.zeros(len(y), dtype=np.float32)
    for group in sorted(set(groups.tolist())):
        train = (groups != group) & action_mask
        valid = groups == group
        if train.sum() < 20 or len(np.unique(y[train])) < 2:
            output[valid] = 0.0
            continue
        model = make_model(model_name)
        model.fit(x[train], y[train])
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x[valid])[:, 1]
        else:
            prob = model.decision_function(x[valid])
        output[valid] = prob.astype(np.float32)
    output[~action_mask] = 0.0
    return output


def make_model(model_name: str):
    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.08),
        )
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.12,
            random_state=20260521,
        )
    raise ValueError(f"Unknown model: {model_name}")


def apply_flips(rows: list[dict[str, str]], flip_mask: np.ndarray) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for row, flip in zip(rows, flip_mask):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        if flip:
            item["fighter"] = "blue" if row["fighter"] == "red" else "red"
        if item["fighter"] != row.get("fighter", ""):
            changed += 1
        output.append(item)
    return output, changed


def print_label_summary(model_name: str, y: np.ndarray, action_mask: np.ndarray, p: np.ndarray) -> None:
    action_y = y[action_mask]
    action_p = p[action_mask]
    pos_p = action_p[action_y == 1]
    neg_p = action_p[action_y == 0]
    print(
        f"label_summary,model={model_name},n={int(action_mask.sum())},"
        f"pos={int(action_y.sum())},pos_rate={float(action_y.mean()) if len(action_y) else 0.0:.4f},"
        f"mean_p={float(action_p.mean()) if len(action_p) else 0.0:.4f},"
        f"pos_mean_p={float(pos_p.mean()) if len(pos_p) else 0.0:.4f},"
        f"neg_mean_p={float(neg_p.mean()) if len(neg_p) else 0.0:.4f}",
        flush=True,
    )


def print_score(label: str, score: dict[str, object], wins: int, n_rows: int) -> None:
    summary = score_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},fighter={summary['fighter']:.6f},"
        f"time={summary['time']:.6f},fp={summary['fp']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
