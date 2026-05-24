#!/usr/bin/env python3
"""Summarize local metric components for prediction CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions


COMPONENTS = [
    ("time", "score_time", 0.50),
    ("fighter", "score_fighter", 0.20),
    ("punch_type", "score_punch_type", 0.10),
    ("effectiveness", "score_effectiveness", 0.08),
    ("hand", "score_hand", 0.06),
    ("target", "score_target", 0.06),
    ("fp_penalty", "fp_penalty", -1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction spec as name=path. Repeat for multiple rows.",
    )
    parser.add_argument(
        "--clear-only",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Score only clear rows. Auto enables this when predictions have a clear column.",
    )
    parser.add_argument("--baseline-name", help="Optional prediction name for delta_macro.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def split_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--predictions must be name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"prediction name is empty in {spec!r}")
    return name, Path(path)


def should_clear_filter(rows: list[dict[str, str]], mode: str) -> bool:
    if mode == "yes":
        return True
    if mode == "no":
        return False
    return bool(rows) and "clear" in rows[0]


def component_means(result: dict[str, object]) -> dict[str, float]:
    by_video = result["by_video"]
    if not isinstance(by_video, dict) or not by_video:
        raise RuntimeError("metric result has no per-video scores")
    means: dict[str, float] = {}
    for out_name, metric_name, _weight in COMPONENTS:
        means[out_name] = sum(float(item[metric_name]) for item in by_video.values()) / len(by_video)
    return means


def weighted_raw(means: dict[str, float]) -> float:
    return sum(means[name] * weight for name, _metric_name, weight in COMPONENTS)


def score_one(
    name: str,
    path: Path,
    gt_rows_all: list[dict[str, str]],
    clear_only_mode: str,
) -> dict[str, object]:
    pred_rows = read_csv_rows(path)
    video_keys = {row["video_key"] for row in pred_rows}
    gt_rows = [row for row in gt_rows_all if row["video_key"] in video_keys]
    clear_only = should_clear_filter(pred_rows, clear_only_mode)
    if clear_only:
        pred_rows = [row for row in pred_rows if row.get("clear") == "true"]
        gt_rows = [row for row in gt_rows if row.get("clear") == "true"]

    result = score_predictions(gt_rows, pred_rows)
    means = component_means(result)
    row: dict[str, object] = {
        "name": name,
        "path": str(path),
        "n_predictions": len(pred_rows),
        "n_videos": len(result["by_video"]),
        "clear_only": str(clear_only).lower(),
    }
    for key, value in means.items():
        row[key] = f"{value:.6f}"
        row[f"weighted_{key}"] = f"{value * component_weight(key):.6f}"
    row["weighted_raw"] = f"{weighted_raw(means):.6f}"
    row["macro_score"] = f"{float(result['macro_score']):.6f}"
    return row


def main() -> int:
    args = parse_args()
    specs = [split_spec(spec) for spec in args.predictions]
    gt_rows = read_csv_rows(args.data_root / "train/punches.csv")
    rows = [score_one(name, path, gt_rows, args.clear_only) for name, path in specs]
    if args.baseline_name:
        baseline_rows = [row for row in rows if row["name"] == args.baseline_name]
        if not baseline_rows:
            raise RuntimeError(f"baseline name not found: {args.baseline_name}")
        baseline = baseline_rows[0]
        for row in rows:
            for name, _metric_name, weight in COMPONENTS:
                delta = float(row[name]) - float(baseline[name])
                row[f"delta_{name}"] = f"{delta:+.6f}"
                row[f"delta_weighted_{name}"] = f"{delta * weight:+.6f}"
            row["delta_weighted_raw"] = f"{float(row['weighted_raw']) - float(baseline['weighted_raw']):+.6f}"
            row["delta_macro"] = f"{float(row['macro_score']) - float(baseline['macro_score']):+.6f}"
    else:
        for row in rows:
            for name, _metric_name, _weight in COMPONENTS:
                row[f"delta_{name}"] = ""
                row[f"delta_weighted_{name}"] = ""
            row["delta_weighted_raw"] = ""
            row["delta_macro"] = ""

    columns = [
        "name",
        "n_predictions",
        "n_videos",
        "clear_only",
    ]
    for name, _metric_name, _weight in COMPONENTS:
        columns.extend([name, f"delta_{name}", f"weighted_{name}", f"delta_weighted_{name}"])
    columns.extend([
        "weighted_raw",
        "delta_weighted_raw",
        "macro_score",
        "delta_macro",
        "path",
    ])
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if args.output:
        write_csv_rows(args.output, rows, columns)
    return 0


def component_weight(name: str) -> float:
    for component_name, _metric_name, weight in COMPONENTS:
        if component_name == name:
            return weight
    raise KeyError(name)


if __name__ == "__main__":
    raise SystemExit(main())
