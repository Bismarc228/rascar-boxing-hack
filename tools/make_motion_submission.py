#!/usr/bin/env python3
"""Generate motion-peak submissions from raw videos.

This is intentionally independent from pose extraction for timing. It uses
downsampled frame differences, then optionally borrows fighter/hand/target from
nearest pose heuristic candidate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from rascar_boxing.validation import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("submissions/motion_peaks.csv"))
    parser.add_argument("--pose-tracks-dir", type=Path)
    parser.add_argument("--count-source", type=Path)
    parser.add_argument("--count-scale", type=float, default=1.0)
    parser.add_argument("--nms-frames", type=int, default=18)
    parser.add_argument("--smooth", type=int, default=7)
    parser.add_argument("--resize-width", type=int, default=160)
    parser.add_argument("--max-per-video", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_punches = read_csv_rows(args.data_root / "train/punches.csv")
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    sample_rows = read_csv_rows(args.data_root / "sample_submission.csv")
    attr_priors = fit_attr_priors(train_punches)
    counts = read_counts(args.count_source) if args.count_source else sample_true_counts(sample_rows)

    selected_by_video: dict[str, list[dict[str, str]]] = {}
    for video in test_videos:
        video_key = video["video_key"]
        target_count = round(counts.get(video_key, 0) * args.count_scale)
        if args.max_per_video is not None:
            target_count = min(target_count, args.max_per_video)
        capacity = sum(1 for row in sample_rows if row["video_key"] == video_key)
        target_count = max(0, min(capacity, target_count))

        scores = motion_scores(args.data_root / video["video_path"], args.resize_width, args.smooth, args.quiet)
        frames = select_peaks(scores, target_count, args.nms_frames)
        pose_candidates = []
        if args.pose_tracks_dir:
            tracks_path = args.pose_tracks_dir / f"{video_key}.jsonl"
            if tracks_path.exists():
                pose_candidates = select_candidates(
                    score_pose_tracks(tracks_path, PoseHeuristicConfig(min_score=0.0)),
                    PoseHeuristicConfig(min_score=0.5, nms_frames=12, count_mode="threshold"),
                    None,
                )

        rows = []
        for frame in frames:
            candidate = nearest_pose(frame, pose_candidates)
            if candidate is None:
                candidate = PunchCandidate(video_key, frame, "blue", "left", "head", 0.0)
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "frame": str(max(0, min(int(video["frame_count"]) - 1, frame))),
                    "fighter": candidate.fighter,
                    "punch_type": attrs["punch_type"],
                    "hand": candidate.hand,
                    "target": candidate.target,
                    "effectiveness": attrs["effectiveness"],
                    "clear": "true",
                }
            )
        selected_by_video[video_key] = rows

    out_rows = fill_sample(sample_rows, selected_by_video)
    write_csv_rows(args.output, out_rows, SUBMISSION_COLUMNS)
    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Wrote {args.output}")
    print(
        "selected="
        + ",".join(f"{key}:{len(selected_by_video.get(key, []))}" for key in sorted(selected_by_video))
    )
    return 0


def motion_scores(video_path: Path, resize_width: int, smooth: int, quiet: bool) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    scores = np.zeros(frame_count, dtype=np.float32)
    prev = None
    pbar = tqdm(
        total=frame_count,
        desc=video_path.name,
        unit="frame",
        mininterval=2.0,
        dynamic_ncols=True,
        disable=quiet or not sys.stderr.isatty(),
    )
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        new_h = max(1, round(h * resize_width / max(1, w)))
        small = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            scores[index] = float(np.percentile(diff, 92))
        prev = gray
        index += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    if index < len(scores):
        scores = scores[:index]
    if smooth > 1 and len(scores) >= smooth:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        scores = np.convolve(scores, kernel, mode="same")
    return scores


def select_peaks(scores: np.ndarray, count: int, nms_frames: int) -> list[int]:
    if count <= 0 or len(scores) == 0:
        return []
    selected: list[int] = []
    for frame in np.argsort(scores)[::-1]:
        frame_i = int(frame)
        if frame_i < 10:
            continue
        if any(abs(frame_i - other) <= nms_frames for other in selected):
            continue
        selected.append(frame_i)
        if len(selected) >= count:
            break
    return sorted(selected)


def nearest_pose(frame: int, candidates: list[PunchCandidate]) -> PunchCandidate | None:
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(item.frame - frame))
    if abs(nearest.frame - frame) > 18:
        return None
    return PunchCandidate(
        nearest.video_key,
        frame,
        nearest.fighter,
        nearest.hand,
        nearest.target,
        nearest.score,
    )


def read_counts(path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    return {
        key: sum(1 for row in rows if row["video_key"] == key and row["clear"] == "true")
        for key in {row["video_key"] for row in rows}
    }


def sample_true_counts(sample_rows: list[dict[str, str]]) -> dict[str, int]:
    keys = {row["video_key"] for row in sample_rows}
    return {
        key: sum(1 for row in sample_rows if row["video_key"] == key and row["clear"] == "true")
        for key in keys
    }


def fill_sample(
    sample_rows: list[dict[str, str]],
    selected_by_video: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    used: dict[str, int] = {}
    output = []
    for sample_row in sample_rows:
        row = {col: sample_row[col] for col in SUBMISSION_COLUMNS}
        video_key = row["video_key"]
        index = used.get(video_key, 0)
        selected = selected_by_video.get(video_key, [])
        if index < len(selected):
            row.update(selected[index])
        else:
            row["clear"] = "false"
        used[video_key] = index + 1
        output.append(row)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
