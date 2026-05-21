#!/usr/bin/env python3
"""Evaluate RGB contact features before final pose candidate selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from tools.evaluate_pose_sequence_spotter import (
    build_rows,
    complete_track_keys,
    fight_group,
)
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_rgb_contact_clip import (
    fit_oof_temporal_heads,
    load_or_extract_features,
    score_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=400)
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--model-name", default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--crop-modes", default="union")
    parser.add_argument("--crop-expand", type=float, default=0.16)
    parser.add_argument("--decode-mode", choices=["sequential", "seek"], default="sequential")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--offset-loss-weight", type=float, default=0.10)
    parser.add_argument("--offset-scale", type=float, default=12.0)
    parser.add_argument("--score-modes", default="blend,prob")
    parser.add_argument("--blend-alphas", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--nms-frames", default="8,10,12")
    parser.add_argument("--cross-nms-frames", default="2,4")
    parser.add_argument("--count-modes", default="root_count,root_rate")
    parser.add_argument("--count-multipliers", default="0.84,0.88,0.92")
    parser.add_argument("--max-shifts", default="0,2,4")
    parser.add_argument("--shift-scales", default="0.0,0.25,0.5")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--write-best-rows", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    ready_keys = complete_track_keys(args.tracks_dir, video_by_key)
    if len(ready_keys) < 3:
        print(f"Need at least 3 complete tracks, got {ready_keys}")
        return 2

    clear_gt = [row for row in punches if row.get("clear") == "true"]
    gt_by_key = group_rows(clear_gt)
    candidates_by_key = build_candidate_pool(args, ready_keys)
    candidates = [candidate for key in ready_keys for candidate in candidates_by_key[key]]
    candidate_rows = candidate_feature_rows(candidates, video_by_key)
    labels, offsets = candidate_labels_offsets(candidates, gt_by_key, args.label_window)
    groups = np.asarray([fight_group(video_by_key[candidate.video_key]) for candidate in candidates])

    print(f"ready={len(ready_keys)} {','.join(ready_keys)}", flush=True)
    print("pool=" + ",".join(f"{key}:{len(candidates_by_key[key])}" for key in ready_keys), flush=True)
    print(
        f"candidate_labels n={len(candidates)} pos={int(labels.sum())} "
        f"pos_rate={float(labels.mean()):.4f} offset_mae={float(np.nanmean(np.abs(offsets))):.3f}",
        flush=True,
    )

    features = load_or_extract_features(args, candidate_rows, video_by_key)
    if len(features) != len(candidates):
        raise ValueError(f"feature count {len(features)} != candidate count {len(candidates)}")
    print(
        f"features shape={features.shape} model={args.model_name} "
        f"clip_len={args.clip_len} stride={args.frame_stride} crop_modes={args.crop_modes}",
        flush=True,
    )
    contact_prob, pred_offsets = fit_oof_temporal_heads(args, features, labels, offsets, groups)
    print_probability_summary(candidates, labels, offsets, contact_prob, pred_offsets)

    gt = [row for row in clear_gt if row["video_key"] in set(ready_keys)]
    train_gt = [row for row in clear_gt if row["video_key"] not in set(ready_keys)]
    train_videos = [row for row in videos if row["video_key"] not in set(ready_keys)]
    train_counts = Counter(row["video_key"] for row in train_gt)
    gt_counts = Counter(row["video_key"] for row in gt)
    attr_priors = fit_attr_priors(train_gt)

    baseline_rows = evaluate_rows(
        args,
        ready_keys,
        video_by_key,
        candidates_by_key,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        score_mode="blend",
        blend_alpha=0.0,
        max_shift=0,
        shift_scale=0.0,
        contact_prob=contact_prob,
        pred_offsets=pred_offsets,
        threshold=0.05,
        nms_frames=10,
        cross_nms=2,
        count_mode="root_count",
        count_multiplier=0.88,
    )
    baseline = score_predictions(gt, baseline_rows)
    baseline_summary = score_summary(baseline)
    print(
        f"baseline_candidate_pool score={baseline['macro_score']:.6f} "
        f"fighter={baseline_summary['fighter']:.6f} time={baseline_summary['time']:.6f} "
        f"fp={baseline_summary['fp']:.6f} n={len(baseline_rows)}",
        flush=True,
    )

    results = grid_results(
        args,
        ready_keys,
        video_by_key,
        candidates_by_key,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        gt,
        baseline,
        contact_prob,
        pred_offsets,
    )
    print(
        "score,delta,fighter,time,fp,wins,n_rows,score_mode,blend_alpha,"
        "threshold,nms,cross_nms,count_mode,count_multiplier,max_shift,shift_scale"
    )
    for item in sorted(results, key=lambda row: row["score"], reverse=True)[: args.top_k]:
        print_result(item, baseline["macro_score"])
    if args.write_best_rows:
        best = max(results, key=lambda row: row["score"])
        rows = best["rows"]
        write_csv_rows(args.write_best_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_best_rows={args.write_best_rows} n_rows={len(rows)}", flush=True)
    return 0


def build_candidate_pool(args: argparse.Namespace, ready_keys: list[str]) -> dict[str, list[PunchCandidate]]:
    output = {}
    for key in ready_keys:
        raw = score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        output[key] = select_candidates(
            raw,
            PoseHeuristicConfig(
                min_score=args.pool_min_score,
                nms_frames=args.pool_nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=0,
            ),
            args.max_candidates_per_video,
        )
    return output


def candidate_feature_rows(
    candidates: list[PunchCandidate],
    video_by_key: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        video = video_by_key[candidate.video_key]
        rows.append(
            {
                "id": str(index),
                "video_id": video["video_id"],
                "agn_index": video["agn_index"],
                "video_key": candidate.video_key,
                "frame": str(candidate.frame),
                "fighter": candidate.fighter,
                "hand": candidate.hand,
                "target": candidate.target,
                "punch_type": "",
                "effectiveness": "",
                "clear": "true",
            }
        )
    return rows


def candidate_labels_offsets(
    candidates: list[PunchCandidate],
    gt_by_key: dict[str, list[dict[str, str]]],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(candidates), dtype=np.float32)
    offsets = np.full(len(candidates), np.nan, dtype=np.float32)
    for index, candidate in enumerate(candidates):
        best: tuple[int, dict[str, str]] | None = None
        for row in gt_by_key.get(candidate.video_key, []):
            if row["fighter"] != candidate.fighter or row["hand"] != candidate.hand:
                continue
            distance = abs(candidate.frame - int(row["frame"]))
            if best is None or distance < best[0]:
                best = (distance, row)
        if best is not None and best[0] <= window:
            labels[index] = 1.0
            offsets[index] = float(int(best[1]["frame"]) - candidate.frame)
    return labels, offsets


def grid_results(
    args: argparse.Namespace,
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    gt: list[dict[str, str]],
    baseline: dict[str, Any],
    contact_prob: np.ndarray,
    pred_offsets: np.ndarray,
) -> list[dict[str, Any]]:
    results = []
    for score_mode in parse_list(args.score_modes):
        alphas = [0.0] if score_mode == "prob" else parse_floats(args.blend_alphas)
        for blend_alpha in alphas:
            for threshold in parse_floats(args.thresholds):
                for nms_frames in parse_ints(args.nms_frames):
                    for cross_nms in parse_ints(args.cross_nms_frames):
                        for count_mode in parse_list(args.count_modes):
                            multipliers = [1.0] if count_mode == "threshold" else parse_floats(args.count_multipliers)
                            for count_multiplier in multipliers:
                                for max_shift in parse_ints(args.max_shifts):
                                    scales = [0.0] if max_shift == 0 else parse_floats(args.shift_scales)
                                    for shift_scale in scales:
                                        rows = evaluate_rows(
                                            args,
                                            ready_keys,
                                            video_by_key,
                                            candidates_by_key,
                                            attr_priors,
                                            train_videos,
                                            train_counts,
                                            gt_counts,
                                            score_mode,
                                            blend_alpha,
                                            max_shift,
                                            shift_scale,
                                            contact_prob,
                                            pred_offsets,
                                            threshold,
                                            nms_frames,
                                            cross_nms,
                                            count_mode,
                                            count_multiplier,
                                        )
                                        score = score_predictions(gt, rows)
                                        summary = score_summary(score)
                                        results.append(
                                            {
                                                "score": score["macro_score"],
                                                "fighter": summary["fighter"],
                                                "time": summary["time"],
                                                "fp": summary["fp"],
                                                "wins": video_wins(score, baseline),
                                                "n_rows": len(rows),
                                                "score_mode": score_mode,
                                                "blend_alpha": blend_alpha,
                                                "threshold": threshold,
                                                "nms_frames": nms_frames,
                                                "cross_nms": cross_nms,
                                                "count_mode": count_mode,
                                                "count_multiplier": count_multiplier,
                                                "max_shift": max_shift,
                                                "shift_scale": shift_scale,
                                                "rows": rows,
                                            }
                                        )
    return results


def evaluate_rows(
    args: argparse.Namespace,
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    score_mode: str,
    blend_alpha: float,
    max_shift: int,
    shift_scale: float,
    contact_prob: np.ndarray,
    pred_offsets: np.ndarray,
    threshold: float,
    nms_frames: int,
    cross_nms: int,
    count_mode: str,
    count_multiplier: float,
) -> list[dict[str, str]]:
    scored = rerank_candidates(
        candidates_by_key,
        contact_prob,
        pred_offsets,
        score_mode,
        blend_alpha,
        max_shift,
        shift_scale,
        video_by_key,
    )
    return build_rows(
        ready_keys,
        video_by_key,
        scored,
        attr_priors,
        train_videos,
        train_counts,
        gt_counts,
        threshold,
        nms_frames,
        cross_nms,
        count_mode,
        count_multiplier,
        pose_prior=0.0,
        snap_window=0,
        stream_scores_by_key_group=None,
    )


def rerank_candidates(
    candidates_by_key: dict[str, list[PunchCandidate]],
    contact_prob: np.ndarray,
    pred_offsets: np.ndarray,
    score_mode: str,
    blend_alpha: float,
    max_shift: int,
    shift_scale: float,
    video_by_key: dict[str, dict[str, str]],
) -> dict[str, list[PunchCandidate]]:
    mean_prob = float(np.mean(contact_prob))
    output: dict[str, list[PunchCandidate]] = {}
    offset = 0
    for key, candidates in candidates_by_key.items():
        rows = []
        frame_count = int(video_by_key[key]["frame_count"])
        for index, candidate in enumerate(candidates):
            prob = float(contact_prob[offset + index])
            if score_mode == "prob":
                score = prob
            elif score_mode == "blend":
                score = float(candidate.score) * max(0.01, 1.0 + blend_alpha * (prob - mean_prob))
            else:
                raise ValueError(f"Unknown score mode: {score_mode}")
            shift = int(round(float(np.clip(pred_offsets[offset + index] * shift_scale, -max_shift, max_shift))))
            frame = max(0, min(frame_count - 1, candidate.frame + shift))
            features = dict(candidate.features)
            features["rgb_contact_prob"] = prob
            features["rgb_contact_offset"] = float(pred_offsets[offset + index])
            rows.append(
                PunchCandidate(
                    candidate.video_key,
                    frame,
                    candidate.fighter,
                    candidate.hand,
                    candidate.target,
                    score,
                    features,
                )
            )
        output[key] = rows
        offset += len(candidates)
    return output


def print_probability_summary(
    candidates: list[PunchCandidate],
    labels: np.ndarray,
    offsets: np.ndarray,
    contact_prob: np.ndarray,
    pred_offsets: np.ndarray,
) -> None:
    by_key: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_key[candidate.video_key].append(index)
    print("video_key,n,pos_rate,mean_p,pos_mean_p,neg_mean_p,target_offset_mae,pred_offset_mae")
    for key, indexes in sorted(by_key.items()):
        idx = np.asarray(indexes, dtype=np.int32)
        y = labels[idx]
        p = contact_prob[idx]
        true = offsets[idx]
        pred = pred_offsets[idx]
        mask = np.isfinite(true)
        pos_p = p[y > 0.5]
        neg_p = p[y <= 0.5]
        print(
            f"{key},{len(indexes)},{float(y.mean()):.4f},{float(p.mean()):.4f},"
            f"{float(pos_p.mean()) if len(pos_p) else 0.0:.4f},"
            f"{float(neg_p.mean()) if len(neg_p) else 0.0:.4f},"
            f"{float(np.mean(np.abs(true[mask]))) if mask.any() else 0.0:.3f},"
            f"{float(np.mean(np.abs(pred[mask] - true[mask]))) if mask.any() else 0.0:.3f}",
            flush=True,
        )


def print_result(item: dict[str, Any], baseline_score: float) -> None:
    print(
        f"{item['score']:.6f},{item['score'] - baseline_score:.6f},"
        f"{item['fighter']:.6f},{item['time']:.6f},{item['fp']:.6f},"
        f"{item['wins']},{item['n_rows']},{item['score_mode']},{item['blend_alpha']},"
        f"{item['threshold']},{item['nms_frames']},{item['cross_nms']},"
        f"{item['count_mode']},{item['count_multiplier']},{item['max_shift']},"
        f"{item['shift_scale']}",
        flush=True,
    )


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_key"]].append(row)
    return dict(grouped)


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
