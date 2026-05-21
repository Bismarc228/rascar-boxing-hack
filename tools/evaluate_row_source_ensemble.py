#!/usr/bin/env python3
"""Evaluate per-video row-source ensemble headroom without brute force."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from tools.evaluate_pose_selection_variants import score_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=Path("data/raw/train/punches.csv"))
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Named row source in name=path format. First source is the base.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [parse_source(item) for item in args.source]
    names = [name for name, _ in sources]
    rows_by_name = {name: read_csv_rows(path) for name, path in sources}
    video_keys = sorted({row["video_key"] for rows in rows_by_name.values() for row in rows})
    gt_rows = [
        row for row in read_csv_rows(args.ground_truth) if row["clear"] == "true" and row["video_key"] in video_keys
    ]

    scores = {}
    for name in names:
        score = score_predictions(gt_rows, rows_by_name[name])
        scores[name] = score
        summary = score_summary(score)
        print(
            f"source={name} score={score['macro_score']:.6f} "
            f"time={summary['time']:.6f} fp={summary['fp_penalty']:.6f} n={len(rows_by_name[name])}"
        )

    base_name = names[0]
    base_rows = rows_by_name[base_name]
    base_score = scores[base_name]["macro_score"]
    oracle_choice = source_oracle_choice(video_keys, scores, names)
    oracle_rows = compose_rows(video_keys, rows_by_name, oracle_choice)
    oracle_score = score_predictions(gt_rows, oracle_rows)
    oracle_summary = score_summary(oracle_score)
    print(
        f"source_oracle score={oracle_score['macro_score']:.6f} "
        f"delta_vs_base={oracle_score['macro_score'] - base_score:.6f} "
        f"time={oracle_summary['time']:.6f} fp={oracle_summary['fp_penalty']:.6f} "
        f"n_rows={len(oracle_rows)}"
    )
    print(
        "source_oracle_choices="
        + ",".join(f"{key}:{name}" for key, name in oracle_choice.items() if name != base_name)
    )
    print_pairwise_oracles(video_keys, rows_by_name, scores, gt_rows, names, base_name, args.top_k)
    print_oracle_by_video(video_keys, scores, names)
    return 0


def parse_source(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--source must be name=path")
    name, path = text.split("=", 1)
    if not name:
        raise ValueError("source name cannot be empty")
    return name, Path(path)


def compose_rows(
    video_keys: list[str],
    rows_by_name: dict[str, list[dict[str, str]]],
    selected: dict[str, str],
) -> list[dict[str, str]]:
    output = []
    for key in video_keys:
        source_name = selected[key]
        output.extend(dict(row) for row in rows_by_name[source_name] if row["video_key"] == key)
    output.sort(key=lambda row: (row["video_key"], int(row["frame"]), row["fighter"], row["hand"]))
    for index, row in enumerate(output, start=1):
        row["id"] = str(index)
    return output


def source_oracle_choice(
    video_keys: list[str],
    scores: dict[str, dict[str, object]],
    names: list[str],
) -> dict[str, str]:
    return {
        key: max(names, key=lambda name: scores[name]["by_video"][key]["final_score"])
        for key in video_keys
    }


def print_pairwise_oracles(
    video_keys: list[str],
    rows_by_name: dict[str, list[dict[str, str]]],
    scores: dict[str, dict[str, object]],
    gt_rows: list[dict[str, str]],
    names: list[str],
    base_name: str,
    top_k: int,
) -> None:
    results = []
    for name in names:
        if name == base_name:
            continue
        choices = {
            key: max([base_name, name], key=lambda source: scores[source]["by_video"][key]["final_score"])
            for key in video_keys
        }
        rows = compose_rows(video_keys, rows_by_name, choices)
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        results.append((score["macro_score"], summary["time"], summary["fp_penalty"], len(rows), name, choices))
    print("pairwise_oracle,score,delta_vs_base,time,fp,n_rows,choices")
    base_score = scores[base_name]["macro_score"]
    for macro, time_score, fp, n_rows, name, choices in sorted(results, reverse=True)[:top_k]:
        choices_text = ",".join(
            f"{key}:{source}" for key, source in choices.items() if source != base_name
        )
        print(
            f"{name},{macro:.6f},{macro - base_score:.6f},"
            f"{time_score:.6f},{fp:.6f},{n_rows},{choices_text}"
        )


def print_oracle_by_video(
    video_keys: list[str],
    scores: dict[str, dict[str, object]],
    names: list[str],
) -> None:
    print("video_key,best_source,best_score," + ",".join(names))
    oracle_values = []
    for key in video_keys:
        values = {name: scores[name]["by_video"][key]["final_score"] for name in names}
        best_name = max(names, key=lambda name: values[name])
        best_score = values[best_name]
        oracle_values.append(best_score)
        print(
            f"{key},{best_name},{best_score:.6f},"
            + ",".join(f"{values[name]:.6f}" for name in names)
        )
    print(f"per_video_source_oracle={float(np.mean(oracle_values)):.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
