#!/usr/bin/env python3
"""Train/evaluate a supervised punch-candidate ranker on cached pose tracks."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

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


FEATURE_NAMES = [
    "log_pose_score",
    "closing",
    "arm_forward",
    "reach",
    "proximity",
    "conf",
    "role_conf",
    "target_dist_norm",
    "head_dist_norm",
    "body_dist_norm",
    "forward_norm",
    "fighter_red",
    "hand_left",
    "target_head",
    "frame_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--max-candidates-per-video", type=int, default=1200)
    parser.add_argument("--label-window", type=int, default=15)
    parser.add_argument("--nms-frames", default="8,12,16")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = []
    for path in sorted(args.tracks_dir.glob("*.jsonl")):
        key = path.stem
        if key not in video_by_key:
            continue
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count == int(video_by_key[key]["frame_count"]):
            ready_keys.append(key)
    if len(ready_keys) < 3:
        print(f"Need at least 3 ready train tracks, got {ready_keys}")
        return 2

    print(f"ready_keys={','.join(ready_keys)}")
    candidates_by_key = {}
    for key in ready_keys:
        all_candidates = score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        candidates_by_key[key] = select_candidates(
            all_candidates,
            PoseHeuristicConfig(min_score=0.05, nms_frames=3),
            args.max_candidates_per_video,
        )
    print("candidate_counts=" + ",".join(f"{k}:{len(v)}" for k, v in candidates_by_key.items()))

    clear_gt = [row for row in punches if row["clear"] == "true"]
    gt_by_key: dict[str, list[dict[str, str]]] = {}
    for row in clear_gt:
        gt_by_key.setdefault(row["video_key"], []).append(row)

    attr_priors = fit_attr_priors(clear_gt)
    results = []
    for heldout in ready_keys:
        train_keys = [key for key in ready_keys if key != heldout]
        train_candidates = [cand for key in train_keys for cand in candidates_by_key[key]]
        X_train = np.array(
            [candidate_features(cand, video_by_key[cand.video_key]) for cand in train_candidates],
            dtype=np.float32,
        )
        y_train = np.array(
            [label_candidate(cand, gt_by_key.get(cand.video_key, []), args.label_window) for cand in train_candidates],
            dtype=np.int8,
        )
        if len(np.unique(y_train)) < 2:
            continue

        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=42,
        )
        model.fit(X_train, y_train)

        heldout_candidates = candidates_by_key[heldout]
        X_eval = np.array(
            [candidate_features(cand, video_by_key[cand.video_key]) for cand in heldout_candidates],
            dtype=np.float32,
        )
        proba = model.predict_proba(X_eval)[:, 1]
        scored = [
            replace_score(cand, float(prob))
            for cand, prob in zip(heldout_candidates, proba)
        ]

        heldout_video = video_by_key[heldout]
        count = len(gt_by_key.get(heldout, []))
        for nms in [int(value) for value in args.nms_frames.split(",") if value]:
            selected = select_candidates(
                scored,
                PoseHeuristicConfig(min_score=0.0, nms_frames=nms),
                count,
            )
            pred_rows = rows_from_candidates([heldout_video], selected, attr_priors)
            gt_rows = gt_by_key.get(heldout, [])
            score = score_predictions(gt_rows, pred_rows)["macro_score"]
            results.append((score, heldout, count, len(selected), nms))

    print("score,video_key,n_gt,n_pred,nms")
    for score, key, n_gt, n_pred, nms in sorted(results, reverse=True)[: args.top_k]:
        print(f"{score:.6f},{key},{n_gt},{n_pred},{nms}")

    by_nms: dict[int, list[float]] = {}
    for score, _, _, _, nms in results:
        by_nms.setdefault(nms, []).append(score)
    print("mean_by_nms")
    for nms, values in sorted(by_nms.items()):
        print(f"{nms},{np.mean(values):.6f},{len(values)}")
    return 0


def candidate_features(candidate: PunchCandidate, video: dict[str, str]) -> list[float]:
    values = candidate.features
    return [
        float(np.log1p(max(0.0, candidate.score))),
        values.get("closing", 0.0),
        values.get("arm_forward", 0.0),
        values.get("reach", 0.0),
        values.get("proximity", 0.0),
        values.get("conf", 0.0),
        values.get("role_conf", 0.0),
        values.get("target_dist_norm", 0.0),
        values.get("head_dist_norm", 0.0),
        values.get("body_dist_norm", 0.0),
        values.get("forward_norm", 0.0) / 1000.0,
        1.0 if candidate.fighter == "red" else 0.0,
        1.0 if candidate.hand == "left" else 0.0,
        1.0 if candidate.target == "head" else 0.0,
        candidate.frame / max(1.0, float(video["frame_count"])),
    ]


def label_candidate(candidate: PunchCandidate, gt_rows: list[dict[str, str]], window: int) -> int:
    best = min((abs(candidate.frame - int(row["frame"])) for row in gt_rows), default=10**9)
    return int(best <= window)


def replace_score(candidate: PunchCandidate, score: float) -> PunchCandidate:
    return PunchCandidate(
        candidate.video_key,
        candidate.frame,
        candidate.fighter,
        candidate.hand,
        candidate.target,
        score,
        candidate.features,
    )


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
                "frame": str(max(0, min(int(video["frame_count"]) - 1, candidate.frame))),
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
