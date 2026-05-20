#!/usr/bin/env python3
"""Evaluate supervised event attributes on cached pose-candidate timings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

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


LABEL_COLUMNS = ["fighter", "punch_type", "hand", "target", "effectiveness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--timing-threshold", type=float, default=0.7)
    parser.add_argument("--timing-nms", type=int, default=8)
    parser.add_argument("--label-window", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    gt_by_key: dict[str, list[dict[str, str]]] = {}
    for row in punches:
        if row["clear"] == "true":
            gt_by_key.setdefault(row["video_key"], []).append(row)

    ready_keys = []
    for path in sorted(args.tracks_dir.glob("*.jsonl")):
        key = path.stem
        if key not in video_by_key:
            continue
        if sum(1 for _ in path.open("r", encoding="utf-8")) == int(video_by_key[key]["frame_count"]):
            ready_keys.append(key)
    print("ready_keys=" + ",".join(ready_keys))

    cand_by_key = {
        key: score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        for key in ready_keys
    }

    baseline_scores = []
    model_scores = []
    for heldout in ready_keys:
        train_examples = []
        train_labels: dict[str, list[str]] = {col: [] for col in LABEL_COLUMNS}
        for key in ready_keys:
            if key == heldout:
                continue
            for candidate in cand_by_key[key]:
                nearest = nearest_gt(candidate, gt_by_key.get(key, []), args.label_window)
                if nearest is None:
                    continue
                train_examples.append(feature_vector(candidate, video_by_key[key]))
                for col in LABEL_COLUMNS:
                    train_labels[col].append(nearest[col])

        if not train_examples:
            continue
        X = np.asarray(train_examples, dtype=np.float32)
        models = {}
        for col in LABEL_COLUMNS:
            encoder = LabelEncoder()
            y = encoder.fit_transform(train_labels[col])
            if len(encoder.classes_) == 1:
                models[col] = (encoder, None)
            else:
                clf = HistGradientBoostingClassifier(
                    max_iter=80,
                    learning_rate=0.06,
                    max_leaf_nodes=12,
                    l2_regularization=0.08,
                    random_state=123,
                )
                clf.fit(X, y)
                models[col] = (encoder, clf)

        selected = select_candidates(
            cand_by_key[heldout],
            PoseHeuristicConfig(min_score=args.timing_threshold, nms_frames=args.timing_nms),
            None,
        )
        attr_priors = fit_attr_priors([row for key in ready_keys if key != heldout for row in gt_by_key.get(key, [])])
        baseline_rows = rows_from_candidates([video_by_key[heldout]], selected, attr_priors)
        model_rows = []
        for index, candidate in enumerate(sorted(selected, key=lambda item: item.frame), start=1):
            video = video_by_key[heldout]
            features = np.asarray([feature_vector(candidate, video)], dtype=np.float32)
            pred = {}
            for col, (encoder, clf) in models.items():
                if clf is None:
                    pred[col] = str(encoder.classes_[0])
                else:
                    pred[col] = str(encoder.inverse_transform(clf.predict(features))[0])
            model_rows.append(
                {
                    "id": str(index),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": heldout,
                    "frame": str(candidate.frame),
                    "fighter": pred["fighter"],
                    "punch_type": pred["punch_type"],
                    "hand": pred["hand"],
                    "target": pred["target"],
                    "effectiveness": pred["effectiveness"],
                    "clear": "true",
                }
            )

        gt = gt_by_key.get(heldout, [])
        base = score_predictions(gt, baseline_rows)["macro_score"]
        model = score_predictions(gt, model_rows)["macro_score"]
        baseline_scores.append(base)
        model_scores.append(model)
        print(f"{heldout},n_gt={len(gt)},n_pred={len(selected)},baseline={base:.6f},model_attrs={model:.6f}")

    if baseline_scores:
        print(f"mean_baseline={float(np.mean(baseline_scores)):.6f}")
        print(f"mean_model_attrs={float(np.mean(model_scores)):.6f}")
    return 0


def nearest_gt(candidate: PunchCandidate, gt_rows: list[dict[str, str]], window: int) -> dict[str, str] | None:
    if not gt_rows:
        return None
    nearest = min(gt_rows, key=lambda row: abs(candidate.frame - int(row["frame"])))
    if abs(candidate.frame - int(nearest["frame"])) > window:
        return None
    return nearest


def feature_vector(candidate: PunchCandidate, video: dict[str, str]) -> list[float]:
    features = candidate.features
    return [
        float(np.log1p(max(0.0, candidate.score))),
        features.get("closing", 0.0),
        features.get("arm_forward", 0.0),
        features.get("reach", 0.0),
        features.get("proximity", 0.0),
        features.get("conf", 0.0),
        features.get("role_conf", 0.0),
        features.get("attacker_red_score", 0.0),
        features.get("attacker_blue_score", 0.0),
        features.get("attacker_color_margin", 0.0),
        features.get("target_dist_norm", 0.0),
        features.get("head_dist_norm", 0.0),
        features.get("body_dist_norm", 0.0),
        features.get("forward_norm", 0.0) / 1000.0,
        1.0 if candidate.fighter == "red" else 0.0,
        1.0 if candidate.hand == "left" else 0.0,
        1.0 if candidate.target == "head" else 0.0,
        candidate.frame / max(1.0, float(video["frame_count"])),
    ]


def rows_from_candidates(
    videos: list[dict[str, str]],
    candidates: list[PunchCandidate],
    attr_priors: dict[str, object],
) -> list[dict[str, str]]:
    video_by_key = {row["video_key"]: row for row in videos}
    rows = []
    for index, candidate in enumerate(sorted(candidates, key=lambda item: item.frame), start=1):
        video = video_by_key[candidate.video_key]
        attrs = estimate_attrs(candidate, attr_priors)
        rows.append(
            {
                "id": str(index),
                "video_id": video["video_id"],
                "agn_index": video["agn_index"],
                "video_key": candidate.video_key,
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


if __name__ == "__main__":
    raise SystemExit(main())
