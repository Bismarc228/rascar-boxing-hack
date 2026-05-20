"""Pose-track punch candidate heuristic."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FPS, SUBMISSION_COLUMNS
from .io import read_csv_rows, write_csv_rows


L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
R_HIP = 12

SIDE_INDEXES = {
    "left": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right": (R_SHOULDER, R_ELBOW, R_WRIST),
}


@dataclass(frozen=True)
class PoseHeuristicConfig:
    kp_conf: float = 0.35
    velocity_lag: int = 3
    min_score: float = 0.08
    nms_frames: int = 12
    count_mode: str = "sample_true"
    count_multiplier: float = 1.0
    capacity_fraction: float = 1.0
    max_predictions: int | None = None
    frame_offset: int = 0


@dataclass(frozen=True)
class PunchCandidate:
    video_key: str
    frame: int
    fighter: str
    hand: str
    target: str
    score: float
    features: dict[str, float] = field(default_factory=dict)


def make_pose_heuristic_submission(
    data_root: Path,
    tracks_dir: Path,
    output_path: Path,
    config: PoseHeuristicConfig,
) -> list[PunchCandidate]:
    train_videos = read_csv_rows(data_root / "train/videos.csv")
    test_videos = read_csv_rows(data_root / "test/videos.csv")
    punches = read_csv_rows(data_root / "train/punches.csv")
    sample_rows = read_csv_rows(data_root / "sample_submission.csv")

    attr_priors = fit_attr_priors(punches)
    selected_by_video: dict[str, list[PunchCandidate]] = {}
    for video in test_videos:
        video_key = video["video_key"]
        tracks_path = tracks_dir / f"{video_key}.jsonl"
        candidates = score_pose_tracks(tracks_path, config)
        count = estimate_count(config, video, train_videos, punches, sample_rows)
        selected_by_video[video_key] = select_candidates(candidates, config, count)

    rows = fill_sample_rows(sample_rows, test_videos, selected_by_video, attr_priors, config.frame_offset)
    write_csv_rows(output_path, rows, SUBMISSION_COLUMNS)
    return [candidate for rows in selected_by_video.values() for candidate in rows]


def predict_for_videos(
    videos: list[dict[str, str]],
    tracks_dir: Path,
    config: PoseHeuristicConfig,
    train_videos: list[dict[str, str]] | None = None,
    train_punches: list[dict[str, str]] | None = None,
    sample_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    train_videos = train_videos or videos
    train_punches = train_punches or []
    rows: list[dict[str, str]] = []
    row_id = 1
    attr_priors = fit_attr_priors(train_punches)

    for video in sorted(videos, key=lambda row: row["video_key"]):
        video_key = video["video_key"]
        tracks_path = tracks_dir / f"{video_key}.jsonl"
        candidates = score_pose_tracks(tracks_path, config)
        count = estimate_count(config, video, train_videos, train_punches, sample_rows or [])
        selected = select_candidates(candidates, config, count)
        for candidate in sorted(selected, key=lambda item: item.frame):
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": video_key,
                    "frame": str(_clamp_frame(candidate.frame + config.frame_offset, video)),
                    "fighter": candidate.fighter,
                    "punch_type": attrs["punch_type"],
                    "hand": candidate.hand,
                    "target": candidate.target,
                    "effectiveness": attrs["effectiveness"],
                    "clear": "true",
                }
            )
            row_id += 1
    return rows


def score_pose_tracks(tracks_path: Path, config: PoseHeuristicConfig) -> list[PunchCandidate]:
    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)

    candidates: list[PunchCandidate] = []
    histories: dict[tuple[str, str], deque[tuple[int, np.ndarray]]] = {
        (role, side): deque(maxlen=max(2, config.velocity_lag + 1))
        for role in ["red", "blue"]
        for side in SIDE_INDEXES
    }

    with tracks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            candidates.extend(_score_record(record, histories, config))
            _update_histories(record, histories, config)
    return candidates


def select_candidates(
    candidates: list[PunchCandidate],
    config: PoseHeuristicConfig,
    count: int | None,
) -> list[PunchCandidate]:
    filtered = [item for item in candidates if item.score >= config.min_score]
    selected: list[PunchCandidate] = []
    for candidate in sorted(filtered, key=lambda item: item.score, reverse=True):
        if count is not None and len(selected) >= count:
            break
        if config.max_predictions is not None and len(selected) >= config.max_predictions:
            break
        if any(abs(candidate.frame - chosen.frame) <= config.nms_frames for chosen in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.frame)


def estimate_count(
    config: PoseHeuristicConfig,
    video: dict[str, str],
    train_videos: list[dict[str, str]],
    train_punches: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
) -> int | None:
    if config.count_mode == "threshold":
        return None
    if config.count_mode == "fixed":
        if config.max_predictions is None:
            raise ValueError("count_mode='fixed' requires max_predictions")
        return config.max_predictions

    video_key = video["video_key"]
    if config.count_mode == "sample_true":
        raw_count = sum(
            1 for row in sample_rows if row["video_key"] == video_key and row["clear"] == "true"
        )
    elif config.count_mode == "sample_capacity_fraction":
        capacity = sum(1 for row in sample_rows if row["video_key"] == video_key)
        raw_count = capacity * config.capacity_fraction
    elif config.count_mode in {"dataset_count", "dataset_rate"}:
        clear_punches = [row for row in train_punches if row.get("clear") == "true"]
        by_video = Counter(row["video_key"] for row in clear_punches)
        videos_by_key = {row["video_key"]: row for row in train_videos}
        peers = [row for row in train_videos if row["dataset_type"] == video["dataset_type"]]
        if not peers:
            peers = train_videos
        if config.count_mode == "dataset_count":
            raw_count = sum(by_video[row["video_key"]] for row in peers) / max(1, len(peers))
        else:
            n_punches = sum(by_video[row["video_key"]] for row in peers)
            n_seconds = sum(int(row["frame_count"]) / FPS for row in peers)
            raw_count = (n_punches / max(1e-6, n_seconds)) * (int(video["frame_count"]) / FPS)
        _ = videos_by_key
    else:
        raise ValueError(f"Unknown count_mode: {config.count_mode}")

    capacity = sum(1 for row in sample_rows if row["video_key"] == video_key) if sample_rows else None
    count = max(0, round(raw_count * config.count_multiplier))
    if capacity is not None:
        count = min(count, capacity)
    return count


def fill_sample_rows(
    sample_rows: list[dict[str, str]],
    videos: list[dict[str, str]],
    selected_by_video: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, Any],
    frame_offset: int = 0,
) -> list[dict[str, str]]:
    videos_by_key = {row["video_key"]: row for row in videos}
    output_rows: list[dict[str, str]] = []
    used_by_video: Counter[str] = Counter()
    for sample_row in sample_rows:
        row = {col: sample_row[col] for col in SUBMISSION_COLUMNS}
        video_key = row["video_key"]
        index = used_by_video[video_key]
        selected = selected_by_video.get(video_key, [])
        if index < len(selected):
            candidate = selected[index]
            attrs = estimate_attrs(candidate, attr_priors)
            row["frame"] = str(_clamp_frame(candidate.frame + frame_offset, videos_by_key[video_key]))
            row["fighter"] = candidate.fighter
            row["punch_type"] = attrs["punch_type"]
            row["hand"] = candidate.hand
            row["target"] = candidate.target
            row["effectiveness"] = attrs["effectiveness"]
            row["clear"] = "true"
        else:
            row["clear"] = "false"
        used_by_video[video_key] += 1
        output_rows.append(row)
    return output_rows


def fit_attr_priors(rows: list[dict[str, str]]) -> dict[str, Any]:
    clear_rows = [row for row in rows if row.get("clear") == "true"] or rows
    defaults = {
        "punch_type": "hook",
        "effectiveness": "landed",
    }
    if not clear_rows:
        return {"global": defaults, "by_target_hand": {}}

    global_priors = {
        "punch_type": Counter(row["punch_type"] for row in clear_rows).most_common(1)[0][0],
        "effectiveness": Counter(row["effectiveness"] for row in clear_rows).most_common(1)[0][0],
    }
    by_target_hand: dict[tuple[str, str], dict[str, str]] = {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in clear_rows:
        grouped[(row["target"], row["hand"])].append(row)
    for key, values in grouped.items():
        by_target_hand[key] = {
            "punch_type": Counter(row["punch_type"] for row in values).most_common(1)[0][0],
            "effectiveness": Counter(row["effectiveness"] for row in values).most_common(1)[0][0],
        }
    return {"global": global_priors, "by_target_hand": by_target_hand}


def estimate_attrs(candidate: PunchCandidate, attr_priors: dict[str, Any]) -> dict[str, str]:
    return attr_priors.get("by_target_hand", {}).get(
        (candidate.target, candidate.hand),
        attr_priors.get("global", {"punch_type": "hook", "effectiveness": "landed"}),
    )


def _score_record(
    record: dict[str, Any],
    histories: dict[tuple[str, str], deque[tuple[int, np.ndarray]]],
    config: PoseHeuristicConfig,
) -> list[PunchCandidate]:
    fighters = record.get("fighters", {})
    if "red" not in fighters or "blue" not in fighters:
        return []

    frame = int(record["frame"])
    video_key = record["video_key"]
    output: list[PunchCandidate] = []
    for role, opponent_role in [("red", "blue"), ("blue", "red")]:
        attacker = fighters[role]
        opponent = fighters[opponent_role]
        attacker_center = _point(attacker.get("center"))
        opponent_center = _point(opponent.get("center"))
        if attacker_center is None or opponent_center is None:
            continue
        forward = opponent_center - attacker_center
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1.0:
            continue
        forward /= forward_norm

        scale = _person_scale(attacker)
        opponent_scale = _person_scale(opponent)
        head_point = _target_point(opponent, "head", config)
        body_point = _target_point(opponent, "body", config)
        if head_point is None or body_point is None:
            continue

        for hand, (shoulder_idx, elbow_idx, wrist_idx) in SIDE_INDEXES.items():
            wrist = _keypoint(attacker, wrist_idx, config.kp_conf)
            elbow = _keypoint(attacker, elbow_idx, config.kp_conf)
            shoulder = _keypoint(attacker, shoulder_idx, config.kp_conf)
            if wrist is None or elbow is None:
                continue

            history = histories[(role, hand)]
            if not history:
                continue
            prev_frame, prev_wrist = history[0]
            dt = max(1, frame - prev_frame)
            velocity = (wrist - prev_wrist) / dt
            closing = max(0.0, float(np.dot(velocity, forward))) / max(1.0, scale * 0.08)
            if closing <= 0:
                continue

            arm_forward = max(0.0, float(np.dot(wrist - elbow, forward))) / max(1.0, scale)
            if shoulder is not None:
                reach = max(0.0, float(np.dot(wrist - shoulder, forward))) / max(1.0, scale)
            else:
                reach = arm_forward

            head_dist = float(np.linalg.norm(wrist - head_point))
            body_dist = float(np.linalg.norm(wrist - body_point))
            target = "head" if head_dist <= body_dist * 0.92 else "body"
            target_dist = min(head_dist, body_dist)
            proximity = math.exp(-0.5 * (target_dist / max(1.0, opponent_scale * 0.9)) ** 2)

            role_conf = float(attacker.get(f"score_{role}", 0.5))
            attacker_red_score = float(attacker.get("score_red", 0.0))
            attacker_blue_score = float(attacker.get("score_blue", 0.0))
            wrist_conf = _keypoint_conf(attacker, wrist_idx)
            elbow_conf = _keypoint_conf(attacker, elbow_idx)
            conf = max(0.0, min(1.0, (wrist_conf + elbow_conf) * 0.5))
            score = closing * (0.35 + arm_forward + 0.35 * reach) * proximity * conf
            score *= 0.75 + 0.25 * max(0.0, min(1.0, role_conf))

            if score > 0:
                output.append(
                    PunchCandidate(
                        video_key,
                        frame,
                        role,
                        hand,
                        target,
                        float(score),
                        {
                            "closing": float(closing),
                            "arm_forward": float(arm_forward),
                            "reach": float(reach),
                            "proximity": float(proximity),
                            "conf": float(conf),
                            "role_conf": float(role_conf),
                            "attacker_red_score": float(attacker_red_score),
                            "attacker_blue_score": float(attacker_blue_score),
                            "attacker_color_margin": float(attacker_red_score - attacker_blue_score),
                            "target_dist_norm": float(target_dist / max(1.0, opponent_scale)),
                            "head_dist_norm": float(head_dist / max(1.0, opponent_scale)),
                            "body_dist_norm": float(body_dist / max(1.0, opponent_scale)),
                            "forward_norm": float(forward_norm),
                        },
                    )
                )
    return output


def _update_histories(
    record: dict[str, Any],
    histories: dict[tuple[str, str], deque[tuple[int, np.ndarray]]],
    config: PoseHeuristicConfig,
) -> None:
    fighters = record.get("fighters", {})
    frame = int(record["frame"])
    for role in ["red", "blue"]:
        fighter = fighters.get(role)
        if not fighter:
            continue
        for side, (_, _, wrist_idx) in SIDE_INDEXES.items():
            wrist = _keypoint(fighter, wrist_idx, config.kp_conf)
            if wrist is not None:
                histories[(role, side)].append((frame, wrist))


def _target_point(fighter: dict[str, Any], target: str, config: PoseHeuristicConfig) -> np.ndarray | None:
    if target == "head":
        points = [
            _keypoint(fighter, index, config.kp_conf * 0.8)
            for index in [0, 1, 2, 3, 4]
        ]
        visible = [point for point in points if point is not None]
        if visible:
            return np.mean(visible, axis=0)
        bbox = fighter.get("bbox")
        if bbox:
            return np.array([(bbox[0] + bbox[2]) * 0.5, bbox[1] + (bbox[3] - bbox[1]) * 0.12])
        return None

    points = [
        _keypoint(fighter, index, config.kp_conf * 0.8)
        for index in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    ]
    visible = [point for point in points if point is not None]
    if visible:
        return np.mean(visible, axis=0)
    bbox = fighter.get("bbox")
    if bbox:
        return np.array([(bbox[0] + bbox[2]) * 0.5, bbox[1] + (bbox[3] - bbox[1]) * 0.52])
    return None


def _person_scale(fighter: dict[str, Any]) -> float:
    bbox = fighter.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    bbox_h = max(1.0, float(bbox[3]) - float(bbox[1]))
    left_shoulder = _keypoint(fighter, L_SHOULDER, 0.1)
    right_shoulder = _keypoint(fighter, R_SHOULDER, 0.1)
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_w = float(np.linalg.norm(left_shoulder - right_shoulder))
    else:
        shoulder_w = 0.0
    return max(30.0, shoulder_w, bbox_h * 0.18)


def _keypoint(fighter: dict[str, Any], index: int, min_conf: float) -> np.ndarray | None:
    keypoints = fighter.get("keypoints") or []
    if index >= len(keypoints):
        return None
    kp = keypoints[index]
    if len(kp) < 3 or float(kp[2]) < min_conf:
        return None
    return np.array([float(kp[0]), float(kp[1])], dtype=float)


def _keypoint_conf(fighter: dict[str, Any], index: int) -> float:
    keypoints = fighter.get("keypoints") or []
    if index >= len(keypoints) or len(keypoints[index]) < 3:
        return 0.0
    return float(keypoints[index][2])


def _point(values: Any) -> np.ndarray | None:
    if not values or len(values) < 2:
        return None
    return np.array([float(values[0]), float(values[1])], dtype=float)


def _clamp_frame(frame: int, video: dict[str, str]) -> int:
    return max(0, min(int(video["frame_count"]) - 1, int(frame)))
