"""Temporal-prior punch baseline.

This is a deliberately simple non-CV baseline: estimate how many clear punches
each video should contain from train metadata, place them at train event-time
quantiles, and fill secondary attributes with train priors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .constants import FPS, SUBMISSION_COLUMNS
from .io import read_csv_rows, write_csv_rows


ATTR_COLUMNS = ["punch_type", "hand", "target", "effectiveness"]


@dataclass(frozen=True)
class TemporalPriorConfig:
    group_cols: tuple[str, ...] = ("dataset_type",)
    count_mode: str = "count"
    spacing: str = "quantile"
    fighter_mode: str = "blue"
    count_multiplier: float = 1.0
    min_predictions: int = 1


@dataclass
class TemporalPrior:
    config: TemporalPriorConfig
    attr_priors: dict[str, str]
    global_rate: float
    global_count: float
    group_rates: dict[tuple[str, ...], float]
    group_counts: dict[tuple[str, ...], float]
    group_times: dict[tuple[str, ...], list[float]]
    global_times: list[float]


def fit_temporal_prior(
    videos: list[dict[str, str]],
    punches: list[dict[str, str]],
    train_video_keys: set[str] | None = None,
    config: TemporalPriorConfig | None = None,
) -> TemporalPrior:
    config = config or TemporalPriorConfig()
    video_by_key = {row["video_key"]: row for row in videos}
    if train_video_keys is None:
        train_video_keys = set(video_by_key)

    clear_punches = [
        row for row in punches if row["clear"] == "true" and row["video_key"] in train_video_keys
    ]
    attr_priors = {
        col: Counter(row[col] for row in clear_punches).most_common(1)[0][0]
        for col in ATTR_COLUMNS
    }

    punch_counts = Counter(row["video_key"] for row in clear_punches)
    grouped_video_stats: dict[tuple[str, ...], list[tuple[int, float]]] = defaultdict(list)
    all_counts: list[int] = []
    all_durations: list[float] = []

    for video_key in sorted(train_video_keys):
        video = video_by_key[video_key]
        count = punch_counts[video_key]
        duration = int(video["frame_count"]) / FPS
        group_key = _group_key(video, config.group_cols)
        grouped_video_stats[group_key].append((count, duration))
        all_counts.append(count)
        all_durations.append(duration)

    group_rates = {
        key: sum(count for count, _ in values) / sum(duration for _, duration in values)
        for key, values in grouped_video_stats.items()
    }
    group_counts = {
        key: mean(count for count, _ in values) for key, values in grouped_video_stats.items()
    }

    group_times: dict[tuple[str, ...], list[float]] = defaultdict(list)
    global_times: list[float] = []
    for punch in clear_punches:
        video = video_by_key[punch["video_key"]]
        frame_count = int(video["frame_count"])
        normalized_time = int(punch["frame"]) / frame_count
        group_times[_group_key(video, config.group_cols)].append(normalized_time)
        global_times.append(normalized_time)

    for values in group_times.values():
        values.sort()
    global_times.sort()

    return TemporalPrior(
        config=config,
        attr_priors=attr_priors,
        global_rate=sum(all_counts) / sum(all_durations),
        global_count=mean(all_counts),
        group_rates=group_rates,
        group_counts=group_counts,
        group_times=dict(group_times),
        global_times=global_times,
    )


def predict_for_videos(
    prior: TemporalPrior,
    videos: list[dict[str, str]],
    capacities: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    row_id = 1
    for video in sorted(videos, key=lambda row: row["video_key"]):
        video_key = video["video_key"]
        frame_count = int(video["frame_count"])
        count = estimate_count(prior, video)
        if capacities is not None:
            count = min(count, capacities.get(video_key, 0))
        count = max(0, count)
        frames = estimate_frames(prior, video, count)

        for index, frame in enumerate(frames):
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": video_key,
                    "frame": str(max(0, min(frame_count - 1, frame))),
                    "fighter": estimate_fighter(prior.config.fighter_mode, index),
                    "punch_type": prior.attr_priors["punch_type"],
                    "hand": prior.attr_priors["hand"],
                    "target": prior.attr_priors["target"],
                    "effectiveness": prior.attr_priors["effectiveness"],
                    "clear": "true",
                }
            )
            row_id += 1
    return rows


def make_temporal_prior_submission(
    data_root: Path,
    output_path: Path,
    config: TemporalPriorConfig,
) -> None:
    train_videos = read_csv_rows(data_root / "train/videos.csv")
    test_videos = read_csv_rows(data_root / "test/videos.csv")
    punches = read_csv_rows(data_root / "train/punches.csv")
    sample_rows = read_csv_rows(data_root / "sample_submission.csv")

    capacities = Counter(row["video_key"] for row in sample_rows)
    prior = fit_temporal_prior(train_videos, punches, config=config)
    predictions = predict_for_videos(prior, test_videos, capacities=dict(capacities))
    predictions_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        predictions_by_video[row["video_key"]].append(row)

    output_rows: list[dict[str, str]] = []
    used_by_video: Counter[str] = Counter()
    for sample_row in sample_rows:
        row = {col: sample_row[col] for col in SUBMISSION_COLUMNS}
        video_key = row["video_key"]
        index = used_by_video[video_key]
        if index < len(predictions_by_video[video_key]):
            pred = predictions_by_video[video_key][index]
            for col in ["frame", "fighter", "punch_type", "hand", "target", "effectiveness", "clear"]:
                row[col] = pred[col]
        else:
            row["clear"] = "false"
        used_by_video[video_key] += 1
        output_rows.append(row)

    write_csv_rows(output_path, output_rows, SUBMISSION_COLUMNS)


def estimate_count(prior: TemporalPrior, video: dict[str, str]) -> int:
    group_key = _group_key(video, prior.config.group_cols)
    if prior.config.count_mode == "rate":
        rate = prior.group_rates.get(group_key, prior.global_rate)
        raw_count = rate * (int(video["frame_count"]) / FPS)
    elif prior.config.count_mode == "count":
        raw_count = prior.group_counts.get(group_key, prior.global_count)
    else:
        raise ValueError(f"Unknown count_mode: {prior.config.count_mode}")
    return max(prior.config.min_predictions, round(raw_count * prior.config.count_multiplier))


def estimate_frames(prior: TemporalPrior, video: dict[str, str], count: int) -> list[int]:
    if count <= 0:
        return []
    frame_count = int(video["frame_count"])
    group_key = _group_key(video, prior.config.group_cols)

    if prior.config.spacing == "uniform":
        return [round((index + 1) * frame_count / (count + 1)) for index in range(count)]
    if prior.config.spacing == "quantile":
        normalized_times = prior.group_times.get(group_key) or prior.global_times
        frames: list[int] = []
        for index in range(count):
            q = (index + 1) / (count + 1)
            time_index = min(len(normalized_times) - 1, max(0, round(q * (len(normalized_times) - 1))))
            frames.append(round(normalized_times[time_index] * frame_count))
        return frames
    raise ValueError(f"Unknown spacing: {prior.config.spacing}")


def estimate_fighter(mode: str, index: int) -> str:
    if mode == "blue":
        return "blue"
    if mode == "red":
        return "red"
    if mode == "alternate":
        return "blue" if index % 2 == 0 else "red"
    raise ValueError(f"Unknown fighter_mode: {mode}")


def _group_key(row: dict[str, str], group_cols: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row[col] for col in group_cols)

