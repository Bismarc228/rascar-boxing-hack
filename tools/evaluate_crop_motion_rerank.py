#!/usr/bin/env python3
"""Evaluate simple crop-motion reranking for pose punch candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=240)
    parser.add_argument("--candidate-min-score", type=float, default=0.05)
    parser.add_argument("--candidate-nms", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=1400)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready = [
        path.stem
        for path in sorted(args.tracks_dir.glob("*.jsonl"))
        if path.stem in video_by_key
        and sum(1 for _ in path.open("r", encoding="utf-8")) == int(video_by_key[path.stem]["frame_count"])
    ]
    print("ready=" + ",".join(ready))
    gt = [row for row in punches if row["clear"] == "true" and row["video_key"] in set(ready)]
    attr_priors = fit_attr_priors([row for row in punches if row["clear"] == "true" and row["video_key"] not in set(ready)])

    base_candidates: dict[str, list[PunchCandidate]] = {}
    records_by_key: dict[str, dict[int, dict[str, object]]] = {}
    motion_by_key: dict[str, dict[int, tuple[float, float, float]]] = {}
    for key in ready:
        candidates = score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        base_candidates[key] = select_candidates(
            candidates,
            PoseHeuristicConfig(min_score=args.candidate_min_score, nms_frames=args.candidate_nms),
            args.top_candidates,
        )
        wanted_frames = {candidate.frame for candidate in base_candidates[key]}
        records_by_key[key] = load_records(args.tracks_dir / f"{key}.jsonl", wanted_frames)
        motion_by_key[key] = compute_motion(
            args.data_root / video_by_key[key]["video_path"],
            records_by_key[key],
            args.resize_width,
            args.quiet,
        )

    for alpha in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.7, 1.0]:
        for beta in [-0.4, 0.0, 0.4]:
            for thr in [0.5, 0.7, 0.9, 1.15]:
                for nms in [8, 12]:
                    rows = []
                    row_id = 1
                    for key in ready:
                        video = video_by_key[key]
                        reranked = [
                            rerank(candidate, motion_by_key[key].get(candidate.frame, (0.0, 0.0, 0.0)), alpha, beta)
                            for candidate in base_candidates[key]
                        ]
                        selected = select_candidates(
                            reranked,
                            PoseHeuristicConfig(min_score=thr, nms_frames=nms),
                            None,
                        )
                        for candidate in selected:
                            attrs = estimate_attrs(candidate, attr_priors)
                            rows.append(
                                {
                                    "id": str(row_id),
                                    "video_id": video["video_id"],
                                    "agn_index": video["agn_index"],
                                    "video_key": key,
                                    "frame": str(candidate.frame),
                                    "fighter": candidate.fighter,
                                    "punch_type": attrs["punch_type"],
                                    "hand": candidate.hand,
                                    "target": candidate.target,
                                    "effectiveness": attrs["effectiveness"],
                                    "clear": "true",
                                }
                            )
                            row_id += 1
                    score = score_predictions(gt, rows)["macro_score"]
                    print(
                        f"score={score:.6f},n={len(rows)},alpha={alpha},beta={beta},thr={thr},nms={nms}",
                        flush=True,
                    )
    return 0


def load_records(path: Path, wanted_frames: set[int]) -> dict[int, dict[str, object]]:
    records = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame"])
            if frame in wanted_frames:
                records[frame] = record
    return records


def compute_motion(
    video_path: Path,
    records: dict[int, dict[str, object]],
    resize_width: int,
    quiet: bool,
) -> dict[int, tuple[float, float, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    wanted = set(records)
    output: dict[int, tuple[float, float, float]] = {}
    prev = None
    pbar = tqdm(total=frame_count, desc=video_path.name, unit="frame", disable=quiet or not sys.stderr.isatty())
    index = 0
    scale = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        new_h = max(1, round(h * resize_width / max(1, w)))
        small = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if scale is None:
            scale = resize_width / max(1, w)
        if prev is not None and index in wanted:
            diff = cv2.absdiff(gray, prev)
            global_motion = float(np.percentile(diff, 92))
            crop_motion = crop_diff(diff, records[index], scale)
            ratio = crop_motion / max(1.0, global_motion)
            output[index] = (crop_motion, global_motion, ratio)
        prev = gray
        index += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return output


def crop_diff(diff: np.ndarray, record: dict[str, object], scale: float) -> float:
    fighters = record.get("fighters", {})
    boxes = []
    if isinstance(fighters, dict):
        for role in ["red", "blue"]:
            fighter = fighters.get(role)
            if isinstance(fighter, dict) and fighter.get("bbox"):
                boxes.append(fighter["bbox"])
    if not boxes:
        return 0.0
    x1 = min(float(box[0]) for box in boxes)
    y1 = min(float(box[1]) for box in boxes)
    x2 = max(float(box[2]) for box in boxes)
    y2 = max(float(box[3]) for box in boxes)
    h, w = diff.shape[:2]
    pad_x = (x2 - x1) * 0.15
    pad_y = (y2 - y1) * 0.12
    ix1 = max(0, min(w - 1, round((x1 - pad_x) * scale)))
    iy1 = max(0, min(h - 1, round((y1 - pad_y) * scale)))
    ix2 = max(ix1 + 1, min(w, round((x2 + pad_x) * scale)))
    iy2 = max(iy1 + 1, min(h, round((y2 + pad_y) * scale)))
    crop = diff[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return 0.0
    return float(np.percentile(crop, 92))


def rerank(
    candidate: PunchCandidate,
    motion: tuple[float, float, float],
    alpha: float,
    beta: float,
) -> PunchCandidate:
    crop_motion, global_motion, ratio = motion
    score = candidate.score * max(0.01, 1.0 + alpha * np.log1p(crop_motion) + beta * np.log1p(ratio))
    features = dict(candidate.features)
    features["crop_motion"] = float(crop_motion)
    features["global_motion"] = float(global_motion)
    features["motion_ratio"] = float(ratio)
    return PunchCandidate(
        candidate.video_key,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
        float(score),
        features,
    )


if __name__ == "__main__":
    raise SystemExit(main())
