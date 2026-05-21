#!/usr/bin/env python3
"""Evaluate fight-level source selection policies from source diagnostic tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "n_pred",
    "pred_per_min",
    "base_count",
    "count_delta_vs_base",
    "base_le15",
    "source_le15",
    "base_med_gap",
    "source_med_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-table", type=Path, required=True)
    parser.add_argument("--test-table", type=Path)
    parser.add_argument("--sources", default="", help="Optional comma-separated source allowlist")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.validation_table)
    sources = parse_sources(args.sources) or sorted({row["source"] for row in rows})
    rows = [row for row in rows if row["source"] in sources and row.get("score")]
    categories = build_categories(rows)
    by_video = group_by_video(rows)
    groups = {key: fight_group(video_rows[0]) for key, video_rows in by_video.items()}
    print_source_summary(rows, sources)
    print_oracle(rows, by_video)
    evaluate_mean_policy(rows, by_video, groups, sources, mode="global")
    evaluate_mean_policy(rows, by_video, groups, sources, mode="root")
    evaluate_regressor_policy(rows, by_video, groups, sources, categories, model_name="ridge")
    evaluate_regressor_policy(rows, by_video, groups, sources, categories, model_name="hgb")
    if args.test_table:
        test_rows = [row for row in read_rows(args.test_table) if row["source"] in sources]
        print_test_choices(rows, test_rows, sources, categories, model_name="ridge")
        print_test_choices(rows, test_rows, sources, categories, model_name="hgb")
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_sources(text: str) -> list[str]:
    return [value for value in text.split(",") if value]


def build_categories(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "source": sorted({row["source"] for row in rows}),
        "data_root": sorted({row["data_root"] for row in rows}),
        "round_number": sorted({row["round_number"] for row in rows}),
    }


def group_by_video(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_key"]].append(row)
    return dict(grouped)


def fight_group(row: dict[str, str]) -> str:
    return "|".join([row["data_root"], row["fight_index"]])


def print_source_summary(rows: list[dict[str, str]], sources: list[str]) -> None:
    print("source,mean_score,n_videos")
    for source in sources:
        values = [float(row["score"]) for row in rows if row["source"] == source]
        if values:
            print(f"{source},{float(np.mean(values)):.6f},{len(values)}")


def print_oracle(
    rows: list[dict[str, str]],
    by_video: dict[str, list[dict[str, str]]],
) -> None:
    choices = {}
    scores = []
    for key, video_rows in by_video.items():
        best = max(video_rows, key=lambda row: float(row["score"]))
        choices[key] = best["source"]
        scores.append(float(best["score"]))
    print(
        f"source_oracle,{float(np.mean(scores)):.6f},choices="
        + ",".join(f"{key}:{source}" for key, source in sorted(choices.items()))
    )


def evaluate_mean_policy(
    rows: list[dict[str, str]],
    by_video: dict[str, list[dict[str, str]]],
    groups: dict[str, str],
    sources: list[str],
    mode: str,
) -> None:
    chosen_scores = []
    choices = {}
    for group in sorted(set(groups.values())):
        valid_keys = [key for key, value in groups.items() if value == group]
        train_rows = [row for row in rows if row["video_key"] not in set(valid_keys)]
        for key in valid_keys:
            video_rows = by_video[key]
            if mode == "root":
                root = video_rows[0]["data_root"]
                candidate_train = [row for row in train_rows if row["data_root"] == root]
                candidate_train = candidate_train or train_rows
            else:
                candidate_train = train_rows
            mean_by_source = {
                source: np.mean([float(row["score"]) for row in candidate_train if row["source"] == source])
                for source in sources
                if any(row["source"] == source for row in candidate_train)
            }
            choice = max(mean_by_source, key=lambda source: mean_by_source[source])
            score = score_for_source(video_rows, choice)
            chosen_scores.append(score)
            choices[key] = choice
    print_policy_result(f"mean_{mode}", chosen_scores, choices)


def evaluate_regressor_policy(
    rows: list[dict[str, str]],
    by_video: dict[str, list[dict[str, str]]],
    groups: dict[str, str],
    sources: list[str],
    categories: dict[str, list[str]],
    model_name: str,
) -> None:
    chosen_scores = []
    choices = {}
    for group in sorted(set(groups.values())):
        valid_keys = [key for key, value in groups.items() if value == group]
        train_rows = [row for row in rows if row["video_key"] not in set(valid_keys)]
        model = fit_model(model_name, train_rows, categories)
        for key in valid_keys:
            video_rows = [row for row in by_video[key] if row["source"] in sources]
            x = features(video_rows, categories)
            pred = model.predict(x)
            choice = video_rows[int(np.argmax(pred))]["source"]
            chosen_scores.append(score_for_source(video_rows, choice))
            choices[key] = choice
    print_policy_result(model_name, chosen_scores, choices)


def print_policy_result(label: str, scores: list[float], choices: dict[str, str]) -> None:
    print(
        f"policy={label} score={float(np.mean(scores)):.6f} choices="
        + ",".join(f"{key}:{source}" for key, source in sorted(choices.items()))
    )


def print_test_choices(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    sources: list[str],
    categories: dict[str, list[str]],
    model_name: str,
) -> None:
    model = fit_model(model_name, train_rows, categories)
    by_video = group_by_video([row for row in test_rows if row["source"] in sources])
    choices = {}
    for key, video_rows in sorted(by_video.items()):
        x = features(video_rows, categories)
        pred = model.predict(x)
        ranked = sorted(
            zip(pred, video_rows),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        choices[key] = ranked[0][1]["source"]
        ranking = ";".join(f"{row['source']}:{float(value):.4f}" for value, row in ranked)
        print(f"test_rank,{model_name},{key},{ranking}")
    print(
        f"test_policy={model_name} choices="
        + ",".join(f"{key}:{source}" for key, source in sorted(choices.items()))
    )


def fit_model(model_name: str, rows: list[dict[str, str]], categories: dict[str, list[str]]):
    x = features(rows, categories)
    y = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    if model_name == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    elif model_name == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=7,
            min_samples_leaf=8,
            l2_regularization=0.5,
            random_state=42,
        )
    else:
        raise ValueError(model_name)
    model.fit(x, y)
    return model


def features(rows: list[dict[str, str]], categories: dict[str, list[str]]) -> np.ndarray:
    return np.stack([feature_row(row, categories) for row in rows]).astype(np.float32)


def feature_row(row: dict[str, str], categories: dict[str, list[str]]) -> np.ndarray:
    values = []
    for name in NUMERIC_FEATURES:
        value = float(row[name]) if row.get(name) not in {"", None} else 0.0
        if name in {"base_med_gap", "source_med_gap"}:
            value = min(value, 200.0) / 200.0
        elif name in {"n_pred", "base_count"}:
            value = value / 150.0
        elif name == "count_delta_vs_base":
            value = value / 50.0
        values.append(value)
    values.append(abs(float(row["count_delta_vs_base"])) / 50.0)
    values.append(float(row["source_le15"]) - float(row["base_le15"]))
    med_delta = float(row["source_med_gap"]) - float(row["base_med_gap"])
    values.append(max(-1.0, min(1.0, med_delta / 200.0)))
    for column in ["source", "data_root", "round_number"]:
        for value in categories[column]:
            values.append(float(row[column] == value))
    return np.asarray(values, dtype=np.float32)


def score_for_source(rows: list[dict[str, str]], source: str) -> float:
    for row in rows:
        if row["source"] == source:
            return float(row["score"])
    raise KeyError(source)


if __name__ == "__main__":
    raise SystemExit(main())
