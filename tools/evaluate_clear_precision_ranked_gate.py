#!/usr/bin/env python3
"""Evaluate precision-ranked fixed-row clear drops for FP/count control."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from tools.evaluate_exchange_state_gate import build_labels, fight_group

try:
    from tools.evaluate_exchange_state_gate import safe_video_wins
except ImportError:
    def safe_video_wins(score: dict[str, object], baseline: dict[str, object], eps: float = 1e-12) -> int:
        wins = 0
        for key, baseline_item in baseline["by_video"].items():
            item = score["by_video"].get(key)
            if item is None and baseline_item["n_gt"] == 0:
                final_score = 1.0
            elif item is None:
                final_score = 0.0
            else:
                final_score = item["final_score"]
            wins += int(final_score > baseline_item["final_score"] + eps)
        return wins


DEFAULT_PREDICTIONS = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_"
    "posehgb094_vitlogreg012_rows_20260522.csv"
)
DEFAULT_SOURCES = [
    "pose_audio_matched=data/processed/diagnostics/"
    "candidate_token_clear_pose_audio_matched_e30_20260522.csv",
    "pose_audio_rgb=data/processed/diagnostics/"
    "candidate_token_clear_pose_audio_rgb_e80_20260522.csv",
    "smoke_debug=data/processed/diagnostics/"
    "candidate_token_clear_smoke_debug_20260522.csv",
]
DEFAULT_THRESHOLDS = (
    "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,"
    "25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,"
    "250,300,400"
)

GUARDS: dict[str, set[str] | None] = {
    "blocked_miss": {"blocked", "miss"},
    "blocked": {"blocked"},
    "miss": {"miss"},
    "all": None,
}
COMPONENTS = {
    "time": "score_time",
    "fighter": "score_fighter",
    "fp_penalty": "fp_penalty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--pkeep-source",
        action="append",
        default=[],
        help="Name/path pair like name=path.csv. Defaults to candidate_token_clear caches.",
    )
    parser.add_argument(
        "--no-default-pkeep-source",
        action="store_true",
        help="Do not load default OOF p_keep caches. Useful for reproducible metadata-only ranker controls.",
    )
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated top-K drop thresholds.")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--output-thresholds",
        type=Path,
        default=Path("data/processed/diagnostics/clear_precision_ranked_thresholds_20260523.csv"),
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("data/processed/diagnostics/clear_precision_ranked_video_20260523.csv"),
    )
    parser.add_argument(
        "--output-ranks",
        type=Path,
        default=Path("data/processed/diagnostics/clear_precision_ranked_rows_20260523.csv"),
    )
    parser.add_argument(
        "--no-oof-ranker",
        action="store_true",
        help="Only evaluate raw p_keep rankers.",
    )
    parser.add_argument(
        "--ranker-feature-set",
        choices=["base", "local"],
        default="base",
        help=(
            "OOF FP ranker features. 'base' uses p_keep sources when present plus row/video metadata; "
            "'local' adds selected-row neighborhood context. With --no-default-pkeep-source, "
            "the same modes become metadata-only controls."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    if not rows:
        raise SystemExit(f"no clear rows found in {args.predictions}")
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    video_keys = {row["video_key"] for row in rows}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in video_keys
    ]
    baseline = score_predictions(gt_rows, rows)
    _, labels_list = build_labels(rows, gt_rows)
    labels = np.asarray(labels_list)
    baseline_summary = score_summary(baseline)

    source_args = args.pkeep_source or ([] if args.no_default_pkeep_source else DEFAULT_SOURCES)
    sources = [parse_source(item) for item in source_args]
    pkeep, match_stats = load_pkeep_sources(rows, sources)
    risk_scores = build_raw_risk_scores(pkeep)
    if not args.no_oof_ranker:
        risk_name = "oof_hgb_fp_pkeep" if pkeep else f"oof_hgb_fp_metadata_{args.ranker_feature_set}"
        risk_scores[risk_name] = fit_oof_fp_ranker(
            rows,
            labels,
            video_by_key,
            pkeep,
            args.ranker_feature_set,
        )

    thresholds = parse_ints(args.thresholds)
    threshold_rows, video_rows = evaluate_rankers(
        rows,
        gt_rows,
        labels,
        baseline,
        baseline_summary,
        risk_scores,
        thresholds,
    )
    rank_rows = build_rank_rows(rows, labels, pkeep, risk_scores)

    write_csv_rows(args.output_thresholds, threshold_rows, threshold_columns())
    write_csv_rows(args.output_video, video_rows, video_columns())
    write_csv_rows(args.output_ranks, rank_rows, rank_columns(pkeep, risk_scores))

    print(
        f"baseline macro={float(baseline['macro_score']):.6f} "
        f"time={baseline_summary['time']:.6f} "
        f"fighter={baseline_summary['fighter']:.6f} "
        f"fp_penalty={baseline_summary['fp_penalty']:.6f} "
        f"rows={len(rows)}"
    )
    print("labels=" + ",".join(f"{name}:{count}" for name, count in Counter(labels).most_common()))
    for name, stats in match_stats.items():
        print(
            f"source={name} exact={stats['exact']} relaxed={stats['relaxed']} "
            f"missing={stats['missing']} duplicate_relaxed={stats['duplicate_relaxed']}"
        )
    print(f"wrote {args.output_thresholds}")
    print(f"wrote {args.output_video}")
    print(f"wrote {args.output_ranks}")
    print_best_rows(threshold_rows, args.top_k)
    return 0


def parse_source(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
    else:
        path = text
        name = Path(path).stem
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_")
    if not safe:
        raise SystemExit(f"invalid pkeep source name in {text!r}")
    return safe, Path(path)


def load_pkeep_sources(
    rows: list[dict[str, str]],
    sources: list[tuple[str, Path]],
) -> tuple[dict[str, np.ndarray], dict[str, Counter[str]]]:
    pkeep: dict[str, np.ndarray] = {}
    stats: dict[str, Counter[str]] = {}
    for name, path in sources:
        source_rows = [row for row in read_csv_rows(path) if row.get("clear") == "true"]
        if not source_rows:
            raise SystemExit(f"no clear rows found in pkeep source {path}")
        if "p_keep" not in source_rows[0]:
            raise SystemExit(f"missing p_keep column in {path}")
        exact: dict[tuple[str, ...], list[float]] = defaultdict(list)
        relaxed: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for row in source_rows:
            value = float(row["p_keep"])
            exact[exact_key(row)].append(value)
            relaxed[relaxed_key(row)].append(value)

        values = []
        counter: Counter[str] = Counter()
        for row in rows:
            key_exact = exact_key(row)
            key_relaxed = relaxed_key(row)
            if key_exact in exact:
                values.append(min(exact[key_exact]))
                counter["exact"] += 1
            elif key_relaxed in relaxed:
                values.append(min(relaxed[key_relaxed]))
                counter["relaxed"] += 1
                if len(relaxed[key_relaxed]) > 1:
                    counter["duplicate_relaxed"] += 1
            else:
                values.append(1.0)
                counter["missing"] += 1
        pkeep[name] = np.asarray(values, dtype=np.float32)
        stats[name] = counter
    return pkeep, stats


def exact_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["video_key"],
        row["frame"],
        row["fighter"],
        row["punch_type"],
        row["hand"],
        row["target"],
        row["effectiveness"],
    )


def relaxed_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["video_key"],
        row["frame"],
        row["fighter"],
        row["hand"],
        row["target"],
    )


def build_raw_risk_scores(pkeep: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    risk = {f"pkeep_{name}": 1.0 - values for name, values in pkeep.items()}
    if len(pkeep) > 1:
        matrix = np.stack(list(pkeep.values()))
        risk["pkeep_min"] = 1.0 - np.min(matrix, axis=0)
        risk["pkeep_mean"] = 1.0 - np.mean(matrix, axis=0)
        risk["pkeep_rankmean"] = np.mean(
            [low_value_rank(values) for values in pkeep.values()],
            axis=0,
        )
    return risk


def low_value_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    if len(values) <= 1:
        ranks.fill(1.0)
        return ranks
    # Low p_keep should rank as high drop risk.
    ranks[order] = np.linspace(1.0, 0.0, len(values), dtype=np.float32)
    return ranks


def fit_oof_fp_ranker(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    video_by_key: dict[str, dict[str, str]],
    pkeep: dict[str, np.ndarray],
    feature_set: str,
) -> np.ndarray:
    x, numeric_count = build_ranker_matrix(rows, video_by_key, pkeep, feature_set)
    numeric_indices = list(range(numeric_count))
    categorical_indices = list(range(numeric_count, x.shape[1]))
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_indices,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ]
    )
    model = make_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=7,
        ),
    )
    y = np.asarray(labels == "fp", dtype=np.int8)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in rows])
    risk = np.zeros(len(rows), dtype=np.float32)
    for group in sorted(set(groups)):
        valid = groups == group
        train = ~valid
        if len(np.unique(y[train])) < 2:
            risk[valid] = float(y[train].mean()) if train.any() else 0.0
            continue
        fold_model = clone(model)
        fold_model.fit(x[train], y[train])
        risk[valid] = fold_model.predict_proba(x[valid])[:, 1].astype(np.float32)
    return risk


def build_ranker_matrix(
    rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    pkeep: dict[str, np.ndarray],
    feature_set: str,
) -> tuple[np.ndarray, int]:
    numeric = []
    categorical = []
    pkeep_names = list(pkeep)
    local_features = (
        build_local_row_features(rows) if feature_set == "local" else [[] for _ in rows]
    )
    for index, row in enumerate(rows):
        video = video_by_key[row["video_key"]]
        frame = as_int(row["frame"], "frame")
        frame_count = max(1, as_int(video["frame_count"], "frame_count"))
        p_values = [float(pkeep[name][index]) for name in pkeep_names]
        values = []
        for value in p_values:
            values.extend([value, 1.0 - value, -np.log10(max(1e-8, value))])
        if p_values:
            values.extend([min(p_values), max(p_values), float(np.mean(p_values))])
        values.extend(
            [
                frame / float(frame_count),
                float(frame_count),
                safe_int(video.get("round_number", "")),
            ]
        )
        values.extend(local_features[index])
        numeric.append(values)
        categorical.append(
            [
                row["effectiveness"],
                row["punch_type"],
                row["fighter"],
                row["hand"],
                row["target"],
                video.get("dataset_type", ""),
                video.get("data_root", ""),
            ]
        )
    numeric_arr = np.asarray(numeric, dtype=np.float32)
    categorical_arr = np.asarray(categorical, dtype=object)
    return np.concatenate([numeric_arr, categorical_arr], axis=1), numeric_arr.shape[1]


def build_local_row_features(rows: list[dict[str, str]]) -> list[list[float]]:
    by_video: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_video[row["video_key"]].append((index, row))
    for video_rows in by_video.values():
        video_rows.sort(key=lambda item: as_int(item[1]["frame"], "frame"))

    output = [[] for _ in rows]
    windows = [5, 10, 15, 30, 60]
    for video_rows in by_video.values():
        for index, row in video_rows:
            frame = as_int(row["frame"], "frame")
            feature_values: list[float] = []
            neighbor_distances = {
                "any": [],
                "same_fighter": [],
                "same_fighter_hand": [],
                "opposite_fighter": [],
                "same_effectiveness": [],
            }
            window_counts = {
                window: Counter()
                for window in windows
            }
            for other_index, other in video_rows:
                if other_index == index:
                    continue
                distance = abs(as_int(other["frame"], "frame") - frame)
                if distance > max(windows):
                    continue
                neighbor_distances["any"].append(distance)
                same_fighter = other["fighter"] == row["fighter"]
                same_hand = other["hand"] == row["hand"]
                same_effectiveness = other["effectiveness"] == row["effectiveness"]
                if same_fighter:
                    neighbor_distances["same_fighter"].append(distance)
                else:
                    neighbor_distances["opposite_fighter"].append(distance)
                if same_fighter and same_hand:
                    neighbor_distances["same_fighter_hand"].append(distance)
                if same_effectiveness:
                    neighbor_distances["same_effectiveness"].append(distance)
                for window in windows:
                    if distance > window:
                        continue
                    counts = window_counts[window]
                    counts["all"] += 1
                    counts["same_fighter"] += int(same_fighter)
                    counts["opposite_fighter"] += int(not same_fighter)
                    counts["same_fighter_hand"] += int(same_fighter and same_hand)
                    counts["same_target"] += int(other["target"] == row["target"])
                    counts["same_type"] += int(other["punch_type"] == row["punch_type"])
                    counts["same_effectiveness"] += int(same_effectiveness)
                    counts[f"eff_{other['effectiveness']}"] += 1
            for window in windows:
                counts = window_counts[window]
                feature_values.extend(
                    [
                        float(counts["all"]),
                        float(counts["same_fighter"]),
                        float(counts["opposite_fighter"]),
                        float(counts["same_fighter_hand"]),
                        float(counts["same_target"]),
                        float(counts["same_type"]),
                        float(counts["same_effectiveness"]),
                        float(counts["eff_landed"]),
                        float(counts["eff_blocked"]),
                        float(counts["eff_miss"]),
                    ]
                )
            for key in [
                "any",
                "same_fighter",
                "same_fighter_hand",
                "opposite_fighter",
                "same_effectiveness",
            ]:
                values = neighbor_distances[key]
                nearest = min(values) if values else 999.0
                feature_values.append(float(min(nearest, 120.0)))
            output[index] = feature_values
    return output


def evaluate_rankers(
    rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    labels: np.ndarray,
    baseline: dict[str, object],
    baseline_summary: dict[str, float],
    risk_scores: dict[str, np.ndarray],
    thresholds: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    threshold_rows = [baseline_row(rows, labels, baseline, baseline_summary)]
    video_rows: list[dict[str, object]] = []
    for model_name, risk in risk_scores.items():
        for guard_name, allowed_effectiveness in GUARDS.items():
            eligible = guard_mask(rows, allowed_effectiveness)
            order = [int(index) for index in np.argsort(-risk, kind="mergesort") if eligible[index]]
            for threshold in thresholds:
                if threshold < 1 or threshold > len(order):
                    continue
                drop_indices = set(order[:threshold])
                keep_mask = np.asarray([index not in drop_indices for index in range(len(rows))], dtype=bool)
                pred_rows = apply_keep_mask(rows, keep_mask)
                score = score_predictions(gt_rows, pred_rows)
                summary = score_summary(score)
                dropped = Counter(str(label) for label, keep in zip(labels, keep_mask) if not keep)
                dropped_by_video = dropped_counts_by_video(rows, labels, keep_mask)
                row = {
                    "model": model_name,
                    "guard": guard_name,
                    "threshold_type": "top_k",
                    "threshold": threshold,
                    "risk_cutoff": f"{float(risk[order[threshold - 1]]):.9f}",
                    "eligible_rows": int(eligible.sum()),
                    "n_kept": len(pred_rows),
                    "n_dropped": int((~keep_mask).sum()),
                    "dropped_fp": dropped.get("fp", 0),
                    "dropped_tp_scorable": dropped.get("tp_scorable", 0),
                    "dropped_tp_time_only": dropped.get("tp_time_only", 0),
                    "fp_to_scorable": ratio(dropped.get("fp", 0), dropped.get("tp_scorable", 0)),
                    "wins": safe_video_wins(score, baseline),
                    "macro_score": f"{float(score['macro_score']):.6f}",
                    "delta_macro": f"{float(score['macro_score']) - float(baseline['macro_score']):+.6f}",
                    "time": f"{summary['time']:.6f}",
                    "delta_time": f"{summary['time'] - baseline_summary['time']:+.6f}",
                    "fighter": f"{summary['fighter']:.6f}",
                    "delta_fighter": f"{summary['fighter'] - baseline_summary['fighter']:+.6f}",
                    "fp_penalty": f"{summary['fp_penalty']:.6f}",
                    "delta_fp_penalty": f"{summary['fp_penalty'] - baseline_summary['fp_penalty']:+.6f}",
                    "min_video_delta": f"{min_video_delta(score, baseline):+.6f}",
                }
                threshold_rows.append(row)
                video_rows.extend(
                    build_video_rows(
                        model_name,
                        guard_name,
                        threshold,
                        score,
                        baseline,
                        dropped_by_video,
                    )
                )
    return threshold_rows, video_rows


def baseline_row(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    baseline: dict[str, object],
    baseline_summary: dict[str, float],
) -> dict[str, object]:
    counts = Counter(str(label) for label in labels)
    return {
        "model": "baseline",
        "guard": "none",
        "threshold_type": "top_k",
        "threshold": 0,
        "risk_cutoff": "",
        "eligible_rows": len(rows),
        "n_kept": len(rows),
        "n_dropped": 0,
        "dropped_fp": 0,
        "dropped_tp_scorable": 0,
        "dropped_tp_time_only": 0,
        "fp_to_scorable": "",
        "wins": 0,
        "macro_score": f"{float(baseline['macro_score']):.6f}",
        "delta_macro": "+0.000000",
        "time": f"{baseline_summary['time']:.6f}",
        "delta_time": "+0.000000",
        "fighter": f"{baseline_summary['fighter']:.6f}",
        "delta_fighter": "+0.000000",
        "fp_penalty": f"{baseline_summary['fp_penalty']:.6f}",
        "delta_fp_penalty": "+0.000000",
        "min_video_delta": "+0.000000",
        "label_fp": counts.get("fp", 0),
        "label_tp_scorable": counts.get("tp_scorable", 0),
        "label_tp_time_only": counts.get("tp_time_only", 0),
    }


def apply_keep_mask(rows: list[dict[str, str]], keep_mask: np.ndarray) -> list[dict[str, str]]:
    return [
        {column: row.get(column, "") for column in SUBMISSION_COLUMNS}
        for row, keep in zip(rows, keep_mask)
        if keep
    ]


def guard_mask(rows: list[dict[str, str]], allowed_effectiveness: set[str] | None) -> np.ndarray:
    if allowed_effectiveness is None:
        return np.ones(len(rows), dtype=bool)
    return np.asarray(
        [row.get("effectiveness", "") in allowed_effectiveness for row in rows],
        dtype=bool,
    )


def score_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        name: float(np.mean([item[field] for item in by_video.values()]))
        for name, field in COMPONENTS.items()
    }


def min_video_delta(score: dict[str, object], baseline: dict[str, object]) -> float:
    deltas = []
    for key, baseline_item in baseline["by_video"].items():
        item = score["by_video"].get(key)
        final_score = float(item["final_score"]) if item is not None else 0.0
        deltas.append(final_score - float(baseline_item["final_score"]))
    return min(deltas) if deltas else 0.0


def dropped_counts_by_video(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    keep_mask: np.ndarray,
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row, label, keep in zip(rows, labels, keep_mask):
        if keep:
            continue
        counts[row["video_key"]][str(label)] += 1
    return counts


def build_video_rows(
    model_name: str,
    guard_name: str,
    threshold: int,
    score: dict[str, object],
    baseline: dict[str, object],
    dropped_by_video: dict[str, Counter[str]],
) -> list[dict[str, object]]:
    rows = []
    for key, baseline_item in sorted(baseline["by_video"].items()):
        item = score["by_video"].get(key)
        if item is None:
            item = empty_video_score(baseline_item)
        dropped = dropped_by_video.get(key, Counter())
        rows.append(
            {
                "model": model_name,
                "guard": guard_name,
                "threshold": threshold,
                "video_key": key,
                "final_score": f"{float(item['final_score']):.6f}",
                "delta_final": f"{float(item['final_score']) - float(baseline_item['final_score']):+.6f}",
                "time": f"{float(item['score_time']):.6f}",
                "delta_time": f"{float(item['score_time']) - float(baseline_item['score_time']):+.6f}",
                "fighter": f"{float(item['score_fighter']):.6f}",
                "delta_fighter": f"{float(item['score_fighter']) - float(baseline_item['score_fighter']):+.6f}",
                "fp_penalty": f"{float(item['fp_penalty']):.6f}",
                "delta_fp_penalty": f"{float(item['fp_penalty']) - float(baseline_item['fp_penalty']):+.6f}",
                "n_pred": item["n_pred"],
                "n_tp": item["n_tp"],
                "n_fp": item["n_fp"],
                "n_fn": item["n_fn"],
                "dropped_fp": dropped.get("fp", 0),
                "dropped_tp_scorable": dropped.get("tp_scorable", 0),
                "dropped_tp_time_only": dropped.get("tp_time_only", 0),
            }
        )
    return rows


def empty_video_score(baseline_item: dict[str, object]) -> dict[str, object]:
    return {
        "final_score": 0.0,
        "score_time": 0.0,
        "score_fighter": 0.0,
        "fp_penalty": 0.0,
        "n_pred": 0,
        "n_tp": 0,
        "n_fp": 0,
        "n_fn": baseline_item["n_gt"],
    }


def build_rank_rows(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    pkeep: dict[str, np.ndarray],
    risk_scores: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    output = []
    for index, (row, label) in enumerate(zip(rows, labels)):
        out: dict[str, object] = {
            "row_index": index,
            "label": str(label),
            "video_key": row["video_key"],
            "frame": row["frame"],
            "fighter": row["fighter"],
            "punch_type": row["punch_type"],
            "hand": row["hand"],
            "target": row["target"],
            "effectiveness": row["effectiveness"],
        }
        for name, values in pkeep.items():
            out[f"pkeep_{name}"] = f"{float(values[index]):.9f}"
        for name, values in risk_scores.items():
            out[f"risk_{name}"] = f"{float(values[index]):.9f}"
        output.append(out)
    return output


def print_best_rows(rows: list[dict[str, object]], top_k: int) -> None:
    candidates = [row for row in rows if row["model"] != "baseline"]
    print("model,guard,threshold,macro_score,delta_macro,delta_fp_penalty,dropped_fp,dropped_tp_scorable,dropped_tp_time_only,fp_to_scorable,min_video_delta")
    for row in sorted(candidates, key=lambda item: float(item["macro_score"]), reverse=True)[:top_k]:
        print(
            ",".join(
                str(row[column])
                for column in [
                    "model",
                    "guard",
                    "threshold",
                    "macro_score",
                    "delta_macro",
                    "delta_fp_penalty",
                    "dropped_fp",
                    "dropped_tp_scorable",
                    "dropped_tp_time_only",
                    "fp_to_scorable",
                    "min_video_delta",
                ]
            )
        )


def threshold_columns() -> list[str]:
    return [
        "model",
        "guard",
        "threshold_type",
        "threshold",
        "risk_cutoff",
        "eligible_rows",
        "n_kept",
        "n_dropped",
        "dropped_fp",
        "dropped_tp_scorable",
        "dropped_tp_time_only",
        "fp_to_scorable",
        "wins",
        "macro_score",
        "delta_macro",
        "time",
        "delta_time",
        "fighter",
        "delta_fighter",
        "fp_penalty",
        "delta_fp_penalty",
        "min_video_delta",
        "label_fp",
        "label_tp_scorable",
        "label_tp_time_only",
    ]


def video_columns() -> list[str]:
    return [
        "model",
        "guard",
        "threshold",
        "video_key",
        "final_score",
        "delta_final",
        "time",
        "delta_time",
        "fighter",
        "delta_fighter",
        "fp_penalty",
        "delta_fp_penalty",
        "n_pred",
        "n_tp",
        "n_fp",
        "n_fn",
        "dropped_fp",
        "dropped_tp_scorable",
        "dropped_tp_time_only",
    ]


def rank_columns(pkeep: dict[str, np.ndarray], risk_scores: dict[str, np.ndarray]) -> list[str]:
    return [
        "row_index",
        "label",
        "video_key",
        "frame",
        "fighter",
        "punch_type",
        "hand",
        "target",
        "effectiveness",
        *[f"pkeep_{name}" for name in pkeep],
        *[f"risk_{name}" for name in risk_scores],
    ]


def ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "inf" if numerator else "0.000000"
    return f"{numerator / denominator:.6f}"


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def safe_int(text: str) -> float:
    try:
        return float(int(text))
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
