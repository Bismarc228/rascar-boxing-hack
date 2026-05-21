#!/usr/bin/env python3
"""Build per-video/fight diagnostics for row-source decisions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import FPS
from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from tools.gate_submission_hybrid import clear_rows_by_video, evaluate_video


FIELDS = [
    "split",
    "video_key",
    "data_root",
    "fight_index",
    "round_number",
    "source",
    "n_pred",
    "pred_per_min",
    "base_count",
    "count_delta_vs_base",
    "base_le15",
    "source_le15",
    "base_med_gap",
    "source_med_gap",
    "n_gt",
    "n_tp",
    "n_fp",
    "n_fn",
    "score",
    "delta_vs_base_score",
    "score_time",
    "score_fighter",
    "fp_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--source", action="append", required=True, help="name=path; first source is base")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [parse_source(value) for value in args.source]
    base_name = sources[0][0]
    rows_by_source = {name: read_csv_rows(path) for name, path in sources}
    videos_path = args.data_root / ("train/videos.csv" if args.split == "validation" else "test/videos.csv")
    video_by_key = {row["video_key"]: row for row in read_csv_rows(videos_path)}
    keys = sorted({row["video_key"] for rows in rows_by_source.values() for row in rows})
    keys = [key for key in keys if key in video_by_key]

    scores_by_source: dict[str, dict[str, Any]] = {}
    if args.split == "validation":
        gt_rows = [
            row
            for row in read_csv_rows(args.data_root / "train/punches.csv")
            if row["clear"] == "true" and row["video_key"] in set(keys)
        ]
        for name, _path in sources:
            scores_by_source[name] = score_predictions(gt_rows, rows_by_source[name])

    base_by_video = clear_rows_by_video(rows_by_source[base_name])
    gate_args = SimpleNamespace(
        max_count_delta=10**9,
        min_base_within_15=0.0,
        min_override_within_15=0.0,
        max_median_gap=10**9,
    )
    out_rows = []
    for key in keys:
        video = video_by_key[key]
        frame_count = int(video["frame_count"])
        for name, _path in sources:
            by_video = clear_rows_by_video(rows_by_source[name])
            gate = evaluate_video(key, base_by_video.get(key, []), by_video.get(key, []), gate_args)
            score = scores_by_source.get(name, {}).get("by_video", {}).get(key, {})
            base_score = scores_by_source.get(base_name, {}).get("by_video", {}).get(key, {})
            n_pred = len(by_video.get(key, []))
            row = {
                "split": args.split,
                "video_key": key,
                "data_root": video.get("data_root", ""),
                "fight_index": video.get("fight_index", ""),
                "round_number": video.get("round_number", ""),
                "source": name,
                "n_pred": n_pred,
                "pred_per_min": f"{n_pred / max(1e-6, frame_count / FPS / 60.0):.4f}",
                "base_count": gate.base_count,
                "count_delta_vs_base": gate.count_delta,
                "base_le15": f"{gate.base_within_15:.4f}",
                "source_le15": f"{gate.override_within_15:.4f}",
                "base_med_gap": f"{gate.base_median_gap:.1f}",
                "source_med_gap": f"{gate.override_median_gap:.1f}",
                "n_gt": score.get("n_gt", ""),
                "n_tp": score.get("n_tp", ""),
                "n_fp": score.get("n_fp", ""),
                "n_fn": score.get("n_fn", ""),
                "score": format_float(score.get("final_score")),
                "delta_vs_base_score": format_float(
                    score.get("final_score", np.nan) - base_score.get("final_score", np.nan)
                    if score and base_score
                    else np.nan
                ),
                "score_time": format_float(score.get("score_time")),
                "score_fighter": format_float(score.get("score_fighter")),
                "fp_penalty": format_float(score.get("fp_penalty")),
            }
            out_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote={args.output}")
    print_source_summary(out_rows, sources, args.split)
    return 0


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--source must be name=path")
    name, path = value.split("=", 1)
    if not name:
        raise ValueError("empty source name")
    return name, Path(path)


def format_float(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        if np.isnan(float(value)):
            return ""
        return f"{float(value):.6f}"
    except Exception:
        return ""


def print_source_summary(
    rows: list[dict[str, Any]],
    sources: list[tuple[str, Path]],
    split: str,
) -> None:
    names = [name for name, _path in sources]
    print("source,n_videos,total_pred,mean_pred_per_min,mean_score,mean_delta_vs_base")
    for name in names:
        items = [row for row in rows if row["source"] == name]
        scores = [float(row["score"]) for row in items if row["score"]]
        deltas = [float(row["delta_vs_base_score"]) for row in items if row["delta_vs_base_score"]]
        total_pred = sum(int(row["n_pred"]) for row in items)
        mean_rate = float(np.mean([float(row["pred_per_min"]) for row in items])) if items else 0.0
        mean_score = f"{float(np.mean(scores)):.6f}" if scores else ""
        mean_delta = f"{float(np.mean(deltas)):.6f}" if deltas else ""
        print(
            f"{name},{len(items)},{total_pred},{mean_rate:.4f},{mean_score},{mean_delta}"
        )
        if split == "validation" and scores:
            print(
                f"summary_delta,{name},{np.mean(deltas):.6f},"
                f"wins_vs_base={sum(delta > 0 for delta in deltas)}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
