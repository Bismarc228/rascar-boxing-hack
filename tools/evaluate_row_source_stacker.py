#!/usr/bin/env python3
"""Evaluate an OOF row-level stacker over saved prediction sources."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from rascar_boxing.constants import EFFECTIVENESS_METRIC_MAP, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from tools.evaluate_pose_selection_variants import score_summary, video_wins

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


ATTR_COLUMNS = ["punch_type", "effectiveness", "hand", "target"]


@dataclass(frozen=True)
class SourceCandidate:
    row: dict[str, str]
    source: str
    row_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Named row source in name=path format. First source is only the default anchor.",
    )
    parser.add_argument("--anchor-source", default="")
    parser.add_argument("--label-window", type=int, default=15)
    parser.add_argument("--match-window", type=int, default=30)
    parser.add_argument("--windows", default="2,4,8,15")
    parser.add_argument("--nms-frames", default="8,10,12")
    parser.add_argument("--cross-nms-frames", default="2,4")
    parser.add_argument("--count-sources", default="")
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--write-best-rows", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [parse_source(item) for item in args.source]
    source_names = [name for name, _ in sources]
    anchor_source = args.anchor_source or source_names[0]
    if anchor_source not in source_names:
        raise ValueError(f"unknown anchor source {anchor_source!r}")
    count_sources = parse_names(args.count_sources) or [anchor_source]
    for name in count_sources:
        if name not in source_names:
            raise ValueError(f"unknown count source {name!r}")

    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}

    rows_by_source = {
        name: [row for row in read_csv_rows(path) if row.get("clear") == "true"]
        for name, path in sources
    }
    video_keys = sorted({row["video_key"] for rows in rows_by_source.values() for row in rows})
    video_set = set(video_keys)
    gt_rows = [
        row
        for row in punches
        if row.get("clear") == "true" and row["video_key"] in video_set
    ]
    gt_by_key = group_rows(gt_rows)

    print_source_scores(gt_rows, rows_by_source)
    candidates_by_key = build_candidates(rows_by_source)
    windows = parse_ints(args.windows)
    groups_by_key = {key: fight_group(video_by_key[key]) for key in video_keys}

    x_by_key: dict[str, np.ndarray] = {}
    y_by_key: dict[str, np.ndarray] = {}
    for key in video_keys:
        x_by_key[key] = build_features(
            candidates_by_key[key],
            candidates_by_key[key],
            video_by_key[key],
            source_names,
            rows_by_source,
            windows,
        )
        y_by_key[key] = build_targets(candidates_by_key[key], gt_by_key.get(key, []), args.label_window)

    scored_by_key: dict[str, np.ndarray] = {}
    for group in sorted(set(groups_by_key.values())):
        valid_keys = [key for key in video_keys if groups_by_key[key] == group]
        train_keys = [key for key in video_keys if groups_by_key[key] != group]
        x_train = np.concatenate([x_by_key[key] for key in train_keys], axis=0)
        y_train = np.concatenate([y_by_key[key] for key in train_keys], axis=0)
        weights = 1.0 + 5.0 * y_train
        model = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=0.2,
            random_state=42,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        for key in valid_keys:
            scored_by_key[key] = np.clip(model.predict(x_by_key[key]), 0.0, 1.0)
        print(
            f"fold={group} train_candidates={len(x_train)} valid={','.join(valid_keys)} "
            f"target_mean={float(np.mean(y_train)):.4f}",
            flush=True,
        )

    results = []
    best_rows: list[dict[str, str]] | None = None
    best_config = None
    for count_source in count_sources:
        counts = Counter(row["video_key"] for row in rows_by_source[count_source])
        for nms in parse_ints(args.nms_frames):
            for cross_nms in parse_ints(args.cross_nms_frames):
                rows = compose_rows(
                    video_keys,
                    video_by_key,
                    candidates_by_key,
                    scored_by_key,
                    counts,
                    nms,
                    cross_nms,
                )
                score = score_predictions(gt_rows, rows)
                summary = score_summary(score)
                base_score = score_predictions(gt_rows, rows_by_source[count_source])
                results.append(
                    (
                        score["macro_score"],
                        summary["time"],
                        summary["fp_penalty"],
                        len(rows),
                        video_wins(score, base_score),
                        count_source,
                        nms,
                        cross_nms,
                        rows,
                    )
                )
                if best_rows is None or score["macro_score"] > best_config[0]:  # type: ignore[index]
                    best_rows = rows
                    best_config = results[-1]

    print("score,time,fp_penalty,n_pred,wins_vs_count_source,count_source,nms,cross_nms")
    for item in sorted(results, reverse=True, key=lambda value: value[0])[: args.top_k]:
        print(
            f"{item[0]:.6f},{item[1]:.6f},{item[2]:.6f},{item[3]},"
            f"{item[4]},{item[5]},{item[6]},{item[7]}"
        )
    if args.write_best_rows and best_rows is not None and best_config is not None:
        write_csv_rows(args.write_best_rows, best_rows, SUBMISSION_COLUMNS)
        print(
            f"wrote_best_rows={args.write_best_rows} score={best_config[0]:.6f} "
            f"count_source={best_config[5]} nms={best_config[6]} cross_nms={best_config[7]}"
        )
    return 0


def parse_source(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--source must be name=path")
    name, path = text.split("=", 1)
    if not name:
        raise ValueError("source name cannot be empty")
    return name, Path(path)


def parse_names(text: str) -> list[str]:
    return [value for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[row["video_key"]].append(row)
    for value in output.values():
        value.sort(key=lambda row: as_int(row["frame"], "frame"))
    return dict(output)


def print_source_scores(gt_rows: list[dict[str, str]], rows_by_source: dict[str, list[dict[str, str]]]) -> None:
    print("source,score,time,fp_penalty,n_rows")
    for name, rows in rows_by_source.items():
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        print(
            f"{name},{score['macro_score']:.6f},{summary['time']:.6f},"
            f"{summary['fp_penalty']:.6f},{len(rows)}"
        )


def build_candidates(rows_by_source: dict[str, list[dict[str, str]]]) -> dict[str, list[SourceCandidate]]:
    output: dict[str, list[SourceCandidate]] = defaultdict(list)
    for source, rows in rows_by_source.items():
        for index, row in enumerate(rows):
            output[row["video_key"]].append(SourceCandidate(row=row, source=source, row_index=index))
    for rows in output.values():
        rows.sort(key=lambda item: (as_int(item.row["frame"], "frame"), item.source, item.row_index))
    return dict(output)


def build_features(
    candidates: list[SourceCandidate],
    video_candidates: list[SourceCandidate],
    video: dict[str, str],
    source_names: list[str],
    rows_by_source: dict[str, list[dict[str, str]]],
    windows: list[int],
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, feature_count(source_names, windows)), dtype=np.float32)

    frames = np.asarray([as_int(item.row["frame"], "frame") for item in video_candidates], dtype=np.int32)
    sources = np.asarray([source_names.index(item.source) for item in video_candidates], dtype=np.int16)
    fighters = np.asarray([0 if item.row["fighter"] == "red" else 1 for item in video_candidates], dtype=np.int8)
    hands = np.asarray([0 if item.row["hand"] == "left" else 1 for item in video_candidates], dtype=np.int8)
    source_counts = {
        name: sum(1 for row in rows_by_source[name] if row["video_key"] == video["video_key"])
        for name in source_names
    }
    frame_count = max(1.0, float(video["frame_count"]))
    rows = []
    for item in candidates:
        row = item.row
        frame = as_int(row["frame"], "frame")
        source_index = source_names.index(item.source)
        values = [
            frame / frame_count,
            float(frame),
            source_counts[item.source] / 150.0,
            1.0 if row["fighter"] == "red" else 0.0,
            1.0 if row["hand"] == "left" else 0.0,
            1.0 if row["target"] == "head" else 0.0,
            1.0 if row["punch_type"] == "hook" else 0.0,
            1.0 if row["punch_type"] == "cross" else 0.0,
            1.0 if row["punch_type"] == "uppercut" else 0.0,
            1.0 if row["effectiveness"] == "landed" else 0.0,
            1.0 if row["effectiveness"] == "blocked" else 0.0,
            1.0 if row["effectiveness"] == "missed" else 0.0,
        ]
        values.extend(1.0 if index == source_index else 0.0 for index in range(len(source_names)))
        for name in source_names:
            values.append(source_counts[name] / 150.0)
            values.append((source_counts[name] - source_counts[item.source]) / 100.0)

        same_fighter = fighters == (0 if row["fighter"] == "red" else 1)
        same_hand = hands == (0 if row["hand"] == "left" else 1)
        frame_delta = np.abs(frames - frame)
        for window in windows:
            near = frame_delta <= window
            near_sources = set(int(value) for value in sources[near])
            same_fh = near & same_fighter & same_hand
            same_f = near & same_fighter
            other_f = near & ~same_fighter
            values.extend(
                [
                    float(np.sum(near)),
                    float(len(near_sources)),
                    float(np.sum(same_fh)),
                    float(np.sum(same_f)),
                    float(np.sum(other_f)),
                    float(np.mean(frame_delta[near]) / max(1, window)) if np.any(near) else 1.0,
                    float(np.min(frame_delta[near]) / max(1, window)) if np.any(near) else 1.0,
                ]
            )
        for source_index_value in range(len(source_names)):
            mask = sources == source_index_value
            if np.any(mask):
                same_mask = mask & same_fighter & same_hand
                source_frame_delta = frame_delta[same_mask] if np.any(same_mask) else frame_delta[mask]
                nearest = float(np.min(source_frame_delta))
            else:
                nearest = 31.0
            values.extend(
                [
                    min(nearest, 31.0) / 31.0,
                    1.0 if nearest <= 4 else 0.0,
                    1.0 if nearest <= 8 else 0.0,
                    1.0 if nearest <= 15 else 0.0,
                ]
            )
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def feature_count(source_names: list[str], windows: list[int]) -> int:
    return 12 + len(source_names) + 2 * len(source_names) + 7 * len(windows) + 4 * len(source_names)


def build_targets(
    candidates: list[SourceCandidate],
    gt_rows: list[dict[str, str]],
    label_window: int,
) -> np.ndarray:
    output = np.zeros(len(candidates), dtype=np.float32)
    if not gt_rows:
        return output
    gt_frames = np.asarray([as_int(row["frame"], "frame") for row in gt_rows], dtype=np.int32)
    for index, item in enumerate(candidates):
        row = item.row
        frame = as_int(row["frame"], "frame")
        diffs = np.abs(gt_frames - frame)
        best_index = int(np.argmin(diffs))
        diff = int(diffs[best_index])
        if diff > label_window:
            continue
        gt = gt_rows[best_index]
        time_score = 1.0 - diff / max(1.0, float(label_window))
        fighter_score = float(row["fighter"] == gt["fighter"])
        hand_score = float(row["hand"] == gt["hand"])
        target_score = float(row["target"] == gt["target"])
        type_score = float(row["punch_type"] == gt["punch_type"])
        eff_score = float(
            EFFECTIVENESS_METRIC_MAP.get(row["effectiveness"], row["effectiveness"])
            == EFFECTIVENESS_METRIC_MAP.get(gt["effectiveness"], gt["effectiveness"])
        )
        output[index] = (
            0.50 * time_score
            + 0.20 * fighter_score
            + 0.10 * type_score
            + 0.08 * eff_score
            + 0.06 * hand_score
            + 0.06 * target_score
        )
    return output


def compose_rows(
    video_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[SourceCandidate]],
    scores_by_key: dict[str, np.ndarray],
    counts_by_key: Counter[str],
    nms_frames: int,
    cross_nms: int,
) -> list[dict[str, str]]:
    output = []
    row_id = 1
    for key in video_keys:
        selected = select_rows(
            candidates_by_key[key],
            scores_by_key[key],
            counts_by_key[key],
            nms_frames,
            cross_nms,
        )
        video = video_by_key[key]
        for item in selected:
            row = {column: item.row[column] for column in SUBMISSION_COLUMNS}
            row["id"] = str(row_id)
            row["video_id"] = video["video_id"]
            row["agn_index"] = video["agn_index"]
            row["video_key"] = key
            row["frame"] = str(max(0, min(int(video["frame_count"]) - 1, as_int(row["frame"], "frame"))))
            row["clear"] = "true"
            output.append(row)
            row_id += 1
    return output


def select_rows(
    candidates: list[SourceCandidate],
    scores: np.ndarray,
    count: int,
    nms_frames: int,
    cross_nms: int,
) -> list[SourceCandidate]:
    selected: list[tuple[SourceCandidate, float]] = []
    order = np.argsort(-scores)
    for index in order:
        if len(selected) >= count:
            break
        item = candidates[int(index)]
        row = item.row
        frame = as_int(row["frame"], "frame")
        group = (row["fighter"], row["hand"])
        suppressed = False
        for chosen, _ in selected:
            chosen_row = chosen.row
            chosen_frame = as_int(chosen_row["frame"], "frame")
            distance = abs(frame - chosen_frame)
            if group == (chosen_row["fighter"], chosen_row["hand"]):
                if distance <= nms_frames:
                    suppressed = True
                    break
            elif distance <= cross_nms:
                suppressed = True
                break
        if not suppressed:
            selected.append((item, float(scores[int(index)])))
    return [item for item, _ in sorted(selected, key=lambda pair: as_int(pair[0].row["frame"], "frame"))]


def fight_group(video: dict[str, str]) -> str:
    return "|".join(
        [
            video["dataset_type"],
            video["data_root"],
            video["fight_index"],
            video["fight_folder"],
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
