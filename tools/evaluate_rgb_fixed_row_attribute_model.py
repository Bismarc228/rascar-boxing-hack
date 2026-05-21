#!/usr/bin/env python3
"""Evaluate cached RGB clip features for fixed-row attribute replacement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, PunchCandidate, score_pose_tracks, select_candidates
from tools.evaluate_pose_selection_variants import video_wins


ATTR_COLUMNS = ["punch_type", "effectiveness", "hand", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--rgb-feature-cache", type=Path, required=True)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=1800)
    parser.add_argument("--candidate-match-window", type=int, default=6)
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--logreg-c", type=float, default=0.35)
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument(
        "--write-variant",
        choices=["punch_type", "effectiveness", "ptype_eff", "hand_target", "all_attrs"],
        default="all_attrs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
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
    print_score("baseline", baseline, len(pred_rows), 0)

    features = load_rgb_features(args.rgb_feature_cache)
    candidates_by_key, feature_by_key = reconstruct_feature_index(args, keys, features)
    x, matched = build_features(args, pred_rows, video_by_key, candidates_by_key, feature_by_key)
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    print(
        f"rgb_feature_rows={len(features)} pred_rows={len(pred_rows)} "
        f"matched={int(matched.sum())} match_rate={float(matched.mean()):.4f} "
        f"x_shape={x.shape}",
        flush=True,
    )

    predictions = oof_attribute_predictions(args, x, labels, groups, pred_rows, matched)
    variants = {
        "punch_type": ["punch_type"],
        "effectiveness": ["effectiveness"],
        "ptype_eff": ["punch_type", "effectiveness"],
        "hand_target": ["hand", "target"],
        "all_attrs": ATTR_COLUMNS,
    }
    print("variant,score,delta,type,effectiveness,hand,target,wins,n_changed")
    rows_by_variant = {}
    for name, columns in variants.items():
        rows, changed = apply_predictions(pred_rows, predictions, columns)
        rows_by_variant[name] = rows
        score = score_predictions(gt_rows, rows)
        summary = attr_summary(score)
        print(
            f"{name},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:.6f},"
            f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
            f"{summary['hand']:.6f},{summary['target']:.6f},"
            f"{video_wins(score, baseline)},{changed}",
            flush=True,
        )
    if args.write_oof_rows:
        write_csv_rows(args.write_oof_rows, rows_by_variant[args.write_variant], SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} "
            f"variant={args.write_variant} n_rows={len(rows_by_variant[args.write_variant])}",
            flush=True,
        )
    print_label_counts(labels)
    return 0


def load_rgb_features(path: Path) -> np.ndarray:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    if features.ndim != 3:
        raise ValueError(f"expected feature tensor [N,T,D], got {features.shape}")
    return features


def reconstruct_feature_index(
    args: argparse.Namespace,
    keys: list[str],
    features: np.ndarray,
) -> tuple[dict[str, list[PunchCandidate]], dict[str, np.ndarray]]:
    candidates_by_key: dict[str, list[PunchCandidate]] = {}
    feature_by_key: dict[str, np.ndarray] = {}
    offset = 0
    for key in keys:
        raw = score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        pool = select_candidates(
            raw,
            PoseHeuristicConfig(
                min_score=args.pool_min_score,
                nms_frames=args.pool_nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=0,
            ),
            args.max_candidates_per_video,
        )
        n_items = len(pool)
        candidates_by_key[key] = pool
        feature_by_key[key] = features[offset : offset + n_items]
        offset += n_items
    if offset != len(features):
        raise ValueError(f"reconstructed {offset} candidates but cache has {len(features)} features")
    return candidates_by_key, feature_by_key


def build_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    feature_by_key: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    matched = []
    dim = next(iter(feature_by_key.values())).shape[-1]
    zero_rgb = np.zeros(dim * 2, dtype=np.float32)
    for row in pred_rows:
        key = row["video_key"]
        index = nearest_candidate_index(row, candidates_by_key[key], args.candidate_match_window)
        if index is None:
            rgb = zero_rgb
            matched.append(False)
        else:
            clip = feature_by_key[key][index]
            rgb = np.concatenate([clip.mean(axis=0), clip.std(axis=0)]).astype(np.float32)
            matched.append(True)
        rows.append(np.concatenate([rgb, row_context_features(row, video_by_key[key])]).astype(np.float32))
    return np.stack(rows), np.asarray(matched, dtype=bool)


def nearest_candidate_index(
    row: dict[str, str],
    candidates: list[PunchCandidate],
    window: int,
) -> int | None:
    frame = as_int(row["frame"], "frame")
    passes = [
        lambda item: item.fighter == row["fighter"] and item.hand == row["hand"],
        lambda item: item.fighter == row["fighter"],
        lambda item: True,
    ]
    for predicate in passes:
        best_index = None
        best_dist = window + 1
        for index, candidate in enumerate(candidates):
            if not predicate(candidate):
                continue
            dist = abs(candidate.frame - frame)
            if dist <= window and dist < best_dist:
                best_index = index
                best_dist = dist
        if best_index is not None:
            return best_index
    return None


def row_context_features(row: dict[str, str], video: dict[str, str]) -> np.ndarray:
    values = [
        1.0 if row["fighter"] == "red" else 0.0,
        1.0 if row["hand"] == "left" else 0.0,
        1.0 if row["target"] == "head" else 0.0,
        1.0 if row["punch_type"] == "jab" else 0.0,
        1.0 if row["punch_type"] == "cross" else 0.0,
        1.0 if row["punch_type"] == "hook" else 0.0,
        1.0 if row["punch_type"] == "uppercut" else 0.0,
        1.0 if row["effectiveness"] == "landed" else 0.0,
        1.0 if row["effectiveness"] == "blocked" else 0.0,
        1.0 if row["effectiveness"] in {"miss", "missed"} else 0.0,
        as_int(row["frame"], "frame") / max(1.0, float(video["frame_count"])),
    ]
    return np.asarray(values, dtype=np.float32)


def matched_attribute_labels(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    label_window: int,
) -> list[dict[str, str] | None]:
    labels: list[dict[str, str] | None] = [None] * len(pred_rows)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    global_indices_by_key: dict[str, list[int]] = {}
    for index, row in enumerate(pred_rows):
        global_indices_by_key.setdefault(row["video_key"], []).append(index)
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            pred = rows[match.pred_index]
            gt = gt_by_key[key][match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) > label_window:
                continue
            labels[global_indices_by_key[key][match.pred_index]] = gt
    return labels


def oof_attribute_predictions(
    args: argparse.Namespace,
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    groups: np.ndarray,
    pred_rows: list[dict[str, str]],
    matched: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {column: np.asarray([row[column] for row in pred_rows], dtype=object) for column in ATTR_COLUMNS}
    trainable = np.asarray([label is not None for label in labels], dtype=bool) & matched
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = (groups != group) & trainable
        valid_predict = valid & matched
        if train.sum() < 20 or valid_predict.sum() == 0:
            print(f"fold={group} train={int(train.sum())} valid={int(valid_predict.sum())} skipped", flush=True)
            continue
        for column in ATTR_COLUMNS:
            values = [labels[index][column] for index in np.where(train)[0]]  # type: ignore[index]
            encoder = LabelEncoder()
            y = encoder.fit_transform(values)
            if len(encoder.classes_) == 1:
                output[column][valid_predict] = encoder.classes_[0]
                continue
            n_components = min(args.pca_components, train.sum() - 1, x.shape[1])
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=42),
                LogisticRegression(
                    C=args.logreg_c,
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=42,
                ),
            )
            model.fit(x[train], y)
            output[column][valid_predict] = encoder.inverse_transform(model.predict(x[valid_predict]))
        print(f"fold={group} train={int(train.sum())} valid={int(valid_predict.sum())}", flush=True)
    return output


def apply_predictions(
    rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    columns: list[str],
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for column in columns:
            item[column] = str(predictions[column][index])
        if any(item[column] != row[column] for column in columns):
            changed += 1
        output.append(item)
    return output, changed


def print_score(label: str, score: dict[str, Any], n_rows: int, wins: int) -> None:
    summary = attr_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},"
        f"type={summary['punch_type']:.6f},eff={summary['effectiveness']:.6f},"
        f"hand={summary['hand']:.6f},target={summary['target']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def attr_summary(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "punch_type": float(np.mean([item["score_punch_type"] for item in by_video.values()])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
    }


def print_label_counts(labels: list[dict[str, str] | None]) -> None:
    usable = [label for label in labels if label is not None]
    print(f"labeled_rows={len(usable)}")
    for column in ATTR_COLUMNS:
        counts: dict[str, int] = {}
        for label in usable:
            counts[label[column]] = counts.get(label[column], 0) + 1
        print("labels_" + column + "=" + ",".join(f"{key}:{value}" for key, value in sorted(counts.items())))


def fight_group(video: dict[str, str]) -> str:
    return "|".join([video["dataset_type"], video["data_root"], video["fight_index"], video["fight_folder"]])


if __name__ == "__main__":
    raise SystemExit(main())
