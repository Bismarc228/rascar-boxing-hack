#!/usr/bin/env python3
"""Run the YOLO-pose role-assignment baseline from the notebook.

The script can resolve a video by `video_key`, write an annotated video, and
cache per-frame red/blue pose tracks as JSONL for later punch heuristics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from rascar_boxing.io import read_csv_rows


L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14

SKELETON_PAIRS = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

LOWER_RED1, UPPER_RED1 = np.array([0, 100, 70]), np.array([12, 255, 255])
LOWER_RED2, UPPER_RED2 = np.array([165, 100, 70]), np.array([180, 255, 255])
LOWER_BLUE, UPPER_BLUE = np.array([90, 100, 70]), np.array([130, 255, 255])
LOWER_WHITE, UPPER_WHITE = np.array([0, 0, 150]), np.array([180, 60, 255])

COLOR_RED_BGR = (0, 0, 255)
COLOR_BLUE_BGR = (255, 50, 0)


def get_polygon_crop_and_mask(frame: np.ndarray, target_kps: list[np.ndarray]) -> tuple[Any, Any]:
    frame_h, frame_w = frame.shape[:2]
    if any(kp[2] < 0.4 for kp in target_kps):
        return None, None

    pts = np.array([[kp[0], kp[1]] for kp in target_kps], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [pts - [x1, y1]], 255)
    return crop, mask


def get_glove_crop_and_mask(
    frame: np.ndarray,
    elbow_kp: np.ndarray,
    wrist_kp: np.ndarray,
) -> tuple[Any, Any]:
    if elbow_kp[2] < 0.4 or wrist_kp[2] < 0.4:
        return None, None

    ex, ey = elbow_kp[:2]
    wx, wy = wrist_kp[:2]
    vx, vy = wx - ex, wy - ey
    gx, gy = int(wx + vx * 0.4), int(wy + vy * 0.4)

    frame_h, frame_w = frame.shape[:2]
    x1, y1 = max(0, gx - 15), max(0, gy - 15)
    x2, y2 = min(frame_w, gx + 15), min(frame_h, gy + 15)
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (gx - x1, gy - y1), 15, 255, -1)
    return crop, mask


def extract_colors(crop: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    roi_pixels = cv2.bitwise_and(hsv, hsv, mask=mask)
    total_pixels = np.count_nonzero(mask)
    if total_pixels == 0:
        return 0.0, 0.0, 0.0

    mask_red = cv2.bitwise_or(
        cv2.inRange(roi_pixels, LOWER_RED1, UPPER_RED1),
        cv2.inRange(roi_pixels, LOWER_RED2, UPPER_RED2),
    )
    mask_blue = cv2.inRange(roi_pixels, LOWER_BLUE, UPPER_BLUE)
    mask_white = cv2.inRange(roi_pixels, LOWER_WHITE, UPPER_WHITE)
    return (
        float(np.sum(mask_red > 0) / total_pixels),
        float(np.sum(mask_blue > 0) / total_pixels),
        float(np.sum(mask_white > 0) / total_pixels),
    )


def classify_person_features(frame: np.ndarray, kps: np.ndarray) -> tuple[float, float, float]:
    total_red, total_blue, total_ref = 0.0, 0.0, 0.0
    weight_sum = 0.0001

    torso_kps = [kps[L_SHOULDER], kps[R_SHOULDER], kps[R_HIP], kps[L_HIP]]
    torso_crop, torso_mask = get_polygon_crop_and_mask(frame, torso_kps)
    if torso_crop is not None:
        conf_w = (sum(kp[2] for kp in torso_kps) / 4) ** 3
        red, blue, white = extract_colors(torso_crop, torso_mask)
        total_red += red * conf_w
        total_blue += blue * conf_w
        total_ref += white * conf_w
        weight_sum += conf_w

    shorts_kps = [kps[L_HIP], kps[R_HIP], kps[R_KNEE], kps[L_KNEE]]
    shorts_crop, shorts_mask = get_polygon_crop_and_mask(frame, shorts_kps)
    if shorts_crop is not None:
        conf_w = (sum(kp[2] for kp in shorts_kps) / 4) ** 3
        red, blue, _ = extract_colors(shorts_crop, shorts_mask)
        total_red += red * conf_w
        total_blue += blue * conf_w
        weight_sum += conf_w

    for elbow_idx, wrist_idx in [(L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)]:
        glove_crop, glove_mask = get_glove_crop_and_mask(frame, kps[elbow_idx], kps[wrist_idx])
        if glove_crop is not None:
            conf_w = ((kps[elbow_idx][2] + kps[wrist_idx][2]) / 2) ** 3 * 0.7
            red, blue, _ = extract_colors(glove_crop, glove_mask)
            total_red += red * conf_w
            total_blue += blue * conf_w
            weight_sum += conf_w

    return total_red / weight_sum, total_blue / weight_sum, total_ref / weight_sum


class BoxingRoleManager:
    def __init__(self) -> None:
        self.prev_centers = {"RED": None, "BLUE": None}

    def update(
        self,
        frame: np.ndarray,
        track_ids: np.ndarray,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        frame_h, frame_w = frame.shape[:2]
        frame_diag = float(np.sqrt(frame_w**2 + frame_h**2))
        candidates = []

        for track_id, bbox, kps in zip(track_ids, bboxes, keypoints):
            if (bbox[3] - bbox[1]) < frame_h * 0.2:
                continue
            red, blue, ref = classify_person_features(frame, kps)
            if ref > 0.15 and red < 0.1 and blue < 0.1:
                continue
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            candidates.append(
                {
                    "tid": int(track_id),
                    "bbox": bbox,
                    "kps": kps,
                    "center": (float(cx), float(cy)),
                    "score_red": red,
                    "score_blue": blue,
                }
            )

        output_fighters: dict[str, dict[str, Any]] = {}
        if not candidates:
            return output_fighters

        cost_matrix = np.zeros((len(candidates), 2))
        for i, candidate in enumerate(candidates):
            score_red = candidate["score_red"]
            score_blue = candidate["score_blue"]

            if self.prev_centers["RED"]:
                dist = np.linalg.norm(np.array(candidate["center"]) - np.array(self.prev_centers["RED"]))
                memory_score = max(0.0, 1.0 - (dist / (frame_diag * 0.3)))
                score_red = score_red * 0.6 + memory_score * 0.4

            if self.prev_centers["BLUE"]:
                dist = np.linalg.norm(np.array(candidate["center"]) - np.array(self.prev_centers["BLUE"]))
                memory_score = max(0.0, 1.0 - (dist / (frame_diag * 0.3)))
                score_blue = score_blue * 0.6 + memory_score * 0.4

            cost_matrix[i, 0] = 1.0 - score_red
            cost_matrix[i, 1] = 1.0 - score_blue

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for row_index, col_index in zip(row_ind, col_ind):
            role = "RED" if col_index == 0 else "BLUE"
            if cost_matrix[row_index, col_index] < 0.85:
                output_fighters[role] = candidates[row_index]
                self.prev_centers[role] = candidates[row_index]["center"]

        if "RED" not in output_fighters:
            self.prev_centers["RED"] = None
        if "BLUE" not in output_fighters:
            self.prev_centers["BLUE"] = None
        return output_fighters


def draw_fighter(frame: np.ndarray, fighter_data: dict[str, Any], color_bgr: tuple[int, int, int], label: str) -> None:
    bbox, kps = fighter_data["bbox"], fighter_data["kps"]
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + 180, y1), color_bgr, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for start_idx, end_idx in SKELETON_PAIRS:
        if start_idx < len(kps) and end_idx < len(kps):
            kp1, kp2 = kps[start_idx], kps[end_idx]
            if kp1[2] > 0.4 and kp2[2] > 0.4:
                cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), color_bgr, 3)
    for kp in kps:
        if kp[2] > 0.4:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 255, 255), -1)


def serialize_fighter(fighter_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": int(fighter_data["tid"]),
        "bbox": [float(x) for x in fighter_data["bbox"]],
        "center": [float(x) for x in fighter_data["center"]],
        "score_red": float(fighter_data["score_red"]),
        "score_blue": float(fighter_data["score_blue"]),
        "keypoints": [[float(v) for v in kp] for kp in fighter_data["kps"]],
    }


def resolve_input(data_root: Path, video_key: str | None, input_path: Path | None) -> tuple[Path, str]:
    if input_path:
        return input_path, video_key or input_path.stem
    if not video_key:
        raise ValueError("Either --input or --video-key is required")

    for rel_csv in ["train/videos.csv", "test/videos.csv"]:
        for row in read_csv_rows(data_root / rel_csv):
            if row["video_key"] == video_key:
                return data_root / row["video_path"], video_key
    raise ValueError(f"Unknown video_key: {video_key}")


def process_video(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    input_path, resolved_key = resolve_input(args.data_root, args.video_key, args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    tracks_out = args.tracks_out or Path("data/processed/pose_tracks") / f"{resolved_key}.jsonl"
    tracks_out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width, height),
        )

    model = YOLO(args.model)
    manager = BoxingRoleManager()
    total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)
    progress = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=resolved_key,
        unit="frame",
        dynamic_ncols=True,
        mininterval=2.0,
        disable=args.no_tqdm or not sys.stderr.isatty(),
    )
    frame_index = 0

    try:
        with tracks_out.open("w", encoding="utf-8") as fh:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                )
                fighters: dict[str, dict[str, Any]] = {}
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    bboxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    keypoints = results[0].keypoints.data.cpu().numpy()
                    fighters = manager.update(frame, track_ids, bboxes, keypoints)

                out_record = {"frame": frame_index, "video_key": resolved_key, "fighters": {}}
                for role, color in [("RED", COLOR_RED_BGR), ("BLUE", COLOR_BLUE_BGR)]:
                    if role in fighters:
                        out_record["fighters"][role.lower()] = serialize_fighter(fighters[role])
                        if writer is not None:
                            draw_fighter(frame, fighters[role], color, role)
                fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                if writer is not None:
                    writer.write(frame)
                frame_index += 1
                progress.update(1)
                if args.progress_every and frame_index % args.progress_every == 0:
                    print(f"processed {frame_index} frames")
    finally:
        progress.close()

    cap.release()
    if writer is not None:
        writer.release()
    print(f"Wrote tracks: {tracks_out}")
    if args.output:
        print(f"Wrote annotated video: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--video-key")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracks-out", type=Path)
    parser.add_argument("--model", default="models/yolo11n-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process_video(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
