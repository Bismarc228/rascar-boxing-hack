#!/usr/bin/env python3
"""Evaluate a per-video clear-count controller on cached pose candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import FPS
from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    estimate_attrs,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from tools.evaluate_pose_model_agreement import (
    agreement_arrays,
    combine_candidates,
    normalized_candidates,
)
from tools.evaluate_pose_selection_variants import estimate_count, score_summary, video_wins


@dataclass
class CountFeaturePack:
    key: str
    x: np.ndarray
    feature_names: list[str]
    base_count: float
    gt_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--primary-tracks-dir", type=Path, required=True)
    parser.add_argument("--secondary-tracks-dir", type=Path)
    parser.add_argument("--mode", choices=["direct_count", "agreement_count"], default="direct_count")
    parser.add_argument("--model", choices=["ridge", "hgb", "mean"], default="ridge")
    parser.add_argument("--target", choices=["count", "log_count", "residual_count"], default="residual_count")
    parser.add_argument("--base-count-prior", default="root_rate")
    parser.add_argument("--base-threshold", type=float, default=0.65)
    parser.add_argument("--nms-frames", type=int, default=8)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--context-feature", default="same_count")
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--agreement-window", type=int, default=4)
    parser.add_argument("--agreement-alpha", type=float, default=0.2)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--secondary-weight", type=float, default=0.8)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-source-video", type=int, default=4000)
    parser.add_argument("--density-thresholds", default="0.4,0.6,0.8,1.0,1.2,1.4")
    parser.add_argument("--count-multipliers", default="0.72,0.78,0.84,0.88,0.92,1.0")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    primary_keys = set(complete_track_keys(args.primary_tracks_dir, video_by_key))
    if args.mode == "agreement_count":
        if args.secondary_tracks_dir is None:
            raise ValueError("--secondary-tracks-dir is required for agreement_count")
        secondary_keys = set(complete_track_keys(args.secondary_tracks_dir, video_by_key))
        ready_keys = sorted(primary_keys & secondary_keys)
    else:
        ready_keys = sorted(primary_keys)
    if not ready_keys:
        print("No complete validation tracks found")
        return 2

    ready_set = set(ready_keys)
    clear_gt = [row for row in punches if row["clear"] == "true"]
    gt = [row for row in clear_gt if row["video_key"] in ready_set]
    gt_counts = Counter(row["video_key"] for row in gt)
    all_counts = Counter(row["video_key"] for row in clear_gt)
    attr_priors = fit_attr_priors([row for row in clear_gt if row["video_key"] not in ready_set])

    candidates_by_key = build_candidate_sets(args, ready_keys)
    print(f"ready={len(ready_keys)} {','.join(ready_keys)}", flush=True)
    print("pool=" + ",".join(f"{key}:{len(candidates_by_key[key])}" for key in ready_keys), flush=True)

    group_by_key = {key: fight_group(video_by_key[key]) for key in ready_keys}
    groups = sorted(set(group_by_key.values()))
    pred_counts: dict[str, int] = {}
    pred_details: dict[str, tuple[int, int, float]] = {}
    density_thresholds = parse_floats(args.density_thresholds)

    for group in groups:
        valid_keys = [key for key in ready_keys if group_by_key[key] == group]
        train_keys = [key for key in ready_keys if group_by_key[key] != group]
        train_packs = [
            build_feature_pack(
                key,
                video_by_key,
                videos,
                all_counts,
                candidates_by_key[key],
                density_thresholds,
                args,
                exclude_keys=set(valid_keys) | {key},
            )
            for key in train_keys
        ]
        valid_packs = [
            build_feature_pack(
                key,
                video_by_key,
                videos,
                all_counts,
                candidates_by_key[key],
                density_thresholds,
                args,
                exclude_keys=set(valid_keys),
            )
            for key in valid_keys
        ]
        model = fit_count_model(train_packs, args)
        for pack in valid_packs:
            pred = predict_count(model, pack, args)
            pred_counts[pack.key] = pred
            pred_details[pack.key] = (pack.gt_count, pred, pack.base_count)
        print(
            f"fold={group} train={len(train_keys)} valid={','.join(valid_keys)} "
            + ",".join(f"{key}:{pred_details[key][0]}->{pred_details[key][1]}" for key in valid_keys),
            flush=True,
        )

    train_videos_for_attrs = [row for row in videos if row["video_key"] not in ready_set]
    train_counts_for_attrs = Counter(row["video_key"] for row in clear_gt if row["video_key"] not in ready_set)

    baseline_results = []
    for mode in ["threshold", "root_rate", "root_count", "root_round_rate", "root_round_count"]:
        mode_multipliers = [1.0] if mode == "threshold" else parse_floats(args.count_multipliers)
        for multiplier in mode_multipliers:
            rows = build_rows_with_policy(
                ready_keys,
                video_by_key,
                candidates_by_key,
                attr_priors,
                train_videos_for_attrs,
                train_counts_for_attrs,
                gt_counts,
                args.base_threshold,
                args.nms_frames,
                args.cross_nms_frames,
                mode,
                multiplier,
            )
            score = score_predictions(gt, rows)
            summary = score_summary(score)
            baseline_results.append(
                (
                    score["macro_score"],
                    summary["time"],
                    summary["fp_penalty"],
                    len(rows),
                    mode,
                    multiplier,
                    score,
                )
            )
    best_baseline = sorted(baseline_results, reverse=True)[0]
    print(
        "best_policy: "
        f"score={best_baseline[0]:.6f},time={best_baseline[1]:.6f},"
        f"fp={best_baseline[2]:.6f},n={best_baseline[3]},"
        f"mode={best_baseline[4]},mult={best_baseline[5]}",
        flush=True,
    )

    controlled_rows = build_rows_with_pred_counts(
        ready_keys,
        video_by_key,
        candidates_by_key,
        attr_priors,
        args.base_threshold,
        args.nms_frames,
        args.cross_nms_frames,
        pred_counts,
    )
    controlled_score = score_predictions(gt, controlled_rows)
    controlled_summary = score_summary(controlled_score)
    print(
        "count_controller: "
        f"score={controlled_score['macro_score']:.6f},time={controlled_summary['time']:.6f},"
        f"fp={controlled_summary['fp_penalty']:.6f},wins={video_wins(controlled_score, best_baseline[6])},"
        f"n={len(controlled_rows)},model={args.model},target={args.target}",
        flush=True,
    )

    oracle_rows = build_rows_with_policy(
        ready_keys,
        video_by_key,
        candidates_by_key,
        attr_priors,
        train_videos_for_attrs,
        train_counts_for_attrs,
        gt_counts,
        args.base_threshold,
        args.nms_frames,
        args.cross_nms_frames,
        "oracle",
        1.0,
    )
    oracle_score = score_predictions(gt, oracle_rows)
    oracle_summary = score_summary(oracle_score)
    print(
        "oracle_count: "
        f"score={oracle_score['macro_score']:.6f},time={oracle_summary['time']:.6f},"
        f"fp={oracle_summary['fp_penalty']:.6f},n={len(oracle_rows)}",
        flush=True,
    )

    print("video,root,gt_count,base_count,pred_count,selected_count,final,fp_penalty")
    selected_counts = Counter(row["video_key"] for row in controlled_rows)
    for key in ready_keys:
        item = controlled_score["by_video"][key]
        gt_count, pred_count, base_count = pred_details[key]
        print(
            f"{key},{video_by_key[key]['data_root']},{gt_count},{base_count:.2f},"
            f"{pred_count},{selected_counts[key]},{item['final_score']:.6f},"
            f"{item['fp_penalty']:.6f}"
        )

    print("top_policies: score,time,fp,n,mode,mult")
    for result in sorted(baseline_results, reverse=True)[: args.top_k]:
        print(
            f"{result[0]:.6f},{result[1]:.6f},{result[2]:.6f},"
            f"{result[3]},{result[4]},{result[5]}"
        )
    return 0


def build_candidate_sets(args: argparse.Namespace, ready_keys: list[str]) -> dict[str, list[PunchCandidate]]:
    if args.mode == "agreement_count":
        assert args.secondary_tracks_dir is not None
        output = {}
        for key in ready_keys:
            primary = normalized_candidates(
                args.primary_tracks_dir / f"{key}.jsonl",
                "primary",
                args.pool_min_score,
                args.pool_nms_frames,
                args.max_candidates_per_source_video,
            )
            secondary = normalized_candidates(
                args.secondary_tracks_dir / f"{key}.jsonl",
                "secondary",
                args.pool_min_score,
                args.pool_nms_frames,
                args.max_candidates_per_source_video,
            )
            agreement = agreement_arrays(primary, secondary, args.agreement_window)
            output[key] = combine_candidates(
                primary,
                secondary,
                agreement,
                args.agreement_alpha,
                args.primary_weight,
                args.secondary_weight,
            )
        return output

    output = {}
    for key in ready_keys:
        candidates = score_pose_tracks(args.primary_tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        candidates = apply_temporal_context(
            candidates,
            PoseHeuristicConfig(
                min_score=0.0,
                context_feature=args.context_feature,
                context_window=args.context_window,
                context_alpha=args.context_alpha,
            ),
        )
        output[key] = candidates
    return output


def build_feature_pack(
    key: str,
    video_by_key: dict[str, dict[str, str]],
    videos: list[dict[str, str]],
    counts: Counter[str],
    candidates: list[PunchCandidate],
    density_thresholds: list[float],
    args: argparse.Namespace,
    exclude_keys: set[str],
) -> CountFeaturePack:
    video = video_by_key[key]
    train_videos = [row for row in videos if row["video_key"] not in exclude_keys]
    train_counts = Counter({row["video_key"]: counts[row["video_key"]] for row in train_videos})
    priors = {
        mode: estimate_count(video, train_videos, train_counts, counts, mode, 1.0) or 0
        for mode in ["root_rate", "dataset_rate", "root_round_rate", "root_count", "root_round_count"]
    }
    base_count = float(priors.get(args.base_count_prior, priors["root_rate"]))

    values: list[float] = []
    names: list[str] = []
    add(values, names, "log_frame_count", np.log1p(int(video["frame_count"])))
    add(values, names, "duration_min", int(video["frame_count"]) / FPS / 60.0)
    add(values, names, "aspect", float(video["width"]) / max(1.0, float(video["height"])))
    add(values, names, "round", float(video["round_number"] or 0))
    for root in ["бокс", "Турнир Бокс", "Турнир Бокс 2"]:
        add(values, names, f"root={root}", float(video["data_root"] == root))
    for dataset_type in ["training", "tournament_1", "tournament_2"]:
        add(values, names, f"dataset={dataset_type}", float(video["dataset_type"] == dataset_type))
    for mode, value in sorted(priors.items()):
        add(values, names, f"prior_{mode}", float(value))

    scores = np.array([candidate.score for candidate in candidates], dtype=np.float32)
    add_density_features(values, names, scores, density_thresholds)
    add_selected_count_features(values, names, candidates, args)
    return CountFeaturePack(
        key=key,
        x=np.asarray(values, dtype=np.float32),
        feature_names=names,
        base_count=base_count,
        gt_count=int(counts[key]),
    )


def add_density_features(
    values: list[float],
    names: list[str],
    scores: np.ndarray,
    thresholds: list[float],
) -> None:
    if len(scores) == 0:
        scores = np.zeros(1, dtype=np.float32)
    for threshold in thresholds:
        add(values, names, f"n_ge_{threshold:g}", float(np.sum(scores >= threshold)))
    for quantile in [0.5, 0.75, 0.9, 0.95, 0.99]:
        add(values, names, f"score_q{quantile:g}", float(np.quantile(scores, quantile)))
    desc = np.sort(scores)[::-1]
    for top_k in [25, 50, 100, 200]:
        add(values, names, f"top{top_k}_sum", float(desc[:top_k].sum()))
    add(values, names, "score_mean", float(scores.mean()))
    add(values, names, "score_std", float(scores.std()))


def add_selected_count_features(
    values: list[float],
    names: list[str],
    candidates: list[PunchCandidate],
    args: argparse.Namespace,
) -> None:
    for multiplier in [0.75, 1.0, 1.25]:
        threshold = args.base_threshold * multiplier
        selected = select_candidates(
            candidates,
            PoseHeuristicConfig(
                min_score=threshold,
                nms_frames=args.nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=args.cross_nms_frames,
            ),
            None,
        )
        add(values, names, f"loose_count_thr_x{multiplier:g}", float(len(selected)))
        if selected:
            frames = np.array([candidate.frame for candidate in selected], dtype=np.float32)
            gaps = np.diff(np.sort(frames))
            add(values, names, f"median_gap_thr_x{multiplier:g}", float(np.median(gaps)) if len(gaps) else 0.0)
        else:
            add(values, names, f"median_gap_thr_x{multiplier:g}", 0.0)


def fit_count_model(packs: list[CountFeaturePack], args: argparse.Namespace):
    x = np.stack([pack.x for pack in packs])
    y_raw = np.array([pack.gt_count for pack in packs], dtype=np.float32)
    base = np.array([pack.base_count for pack in packs], dtype=np.float32)
    if args.target == "residual_count":
        y = y_raw - base
    elif args.target == "log_count":
        y = np.log1p(y_raw)
    else:
        y = y_raw

    if args.model == "mean":
        return ("mean", float(np.mean(y)))
    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            min_samples_leaf=3,
            random_state=42,
        )
    else:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
    model.fit(x, y)
    return model


def predict_count(model, pack: CountFeaturePack, args: argparse.Namespace) -> int:
    if isinstance(model, tuple):
        pred = model[1]
    else:
        pred = float(model.predict(pack.x[None])[0])
    if args.target == "residual_count":
        count = pack.base_count + pred
    elif args.target == "log_count":
        count = np.expm1(pred)
    else:
        count = pred
    return max(0, int(round(float(count))))


def build_rows_with_policy(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    train_videos: list[dict[str, str]],
    train_counts: Counter[str],
    gt_counts: Counter[str],
    threshold: float,
    nms_frames: int,
    cross_nms: int,
    count_mode: str,
    count_multiplier: float,
) -> list[dict[str, str]]:
    pred_counts = {
        key: estimate_count(
            video_by_key[key],
            train_videos,
            train_counts,
            gt_counts,
            count_mode,
            count_multiplier,
        )
        for key in ready_keys
    }
    return build_rows_with_pred_counts(
        ready_keys,
        video_by_key,
        candidates_by_key,
        attr_priors,
        threshold,
        nms_frames,
        cross_nms,
        pred_counts,
    )


def build_rows_with_pred_counts(
    ready_keys: list[str],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    attr_priors: dict[str, object],
    threshold: float,
    nms_frames: int,
    cross_nms: int,
    pred_counts: dict[str, int | None],
) -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for key in sorted(ready_keys):
        video = video_by_key[key]
        selected = select_candidates(
            candidates_by_key[key],
            PoseHeuristicConfig(
                min_score=threshold,
                nms_frames=nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=cross_nms,
            ),
            pred_counts[key],
        )
        for candidate in selected:
            attrs = estimate_attrs(candidate, attr_priors)
            rows.append(
                {
                    "id": str(row_id),
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": key,
                    "frame": str(max(0, min(int(video["frame_count"]) - 1, candidate.frame))),
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


def complete_track_keys(tracks_dir: Path, video_by_key: dict[str, dict[str, str]]) -> list[str]:
    keys = []
    for path in sorted(tracks_dir.glob("*.jsonl")):
        key = path.stem
        if key not in video_by_key:
            continue
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count == int(video_by_key[key]["frame_count"]):
            keys.append(key)
    return keys


def fight_group(video: dict[str, str]) -> str:
    return "|".join(
        [
            video["dataset_type"],
            video["data_root"],
            video["fight_index"],
            video["fight_folder"],
        ]
    )


def add(values: list[float], names: list[str], name: str, value: float) -> None:
    names.append(name)
    values.append(float(value))


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


if __name__ == "__main__":
    raise SystemExit(main())
