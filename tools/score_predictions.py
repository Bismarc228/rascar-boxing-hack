#!/usr/bin/env python3
"""Score predictions against labeled punch rows with the local metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=Path("data/raw/train/punches.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_rows = read_csv_rows(args.ground_truth)
    pred_rows = read_csv_rows(args.predictions)
    result = score_predictions(gt_rows, pred_rows)

    print(f"macro_score={result['macro_score']:.6f}")
    print("video_key,n_gt,n_pred,n_tp,n_fp,n_fn,final_score,time,fighter,type,effectiveness,hand,target,fp_penalty")
    for video_key, item in result["by_video"].items():
        print(
            ",".join(
                [
                    video_key,
                    str(item["n_gt"]),
                    str(item["n_pred"]),
                    str(item["n_tp"]),
                    str(item["n_fp"]),
                    str(item["n_fn"]),
                    f"{item['final_score']:.6f}",
                    f"{item['score_time']:.6f}",
                    f"{item['score_fighter']:.6f}",
                    f"{item['score_punch_type']:.6f}",
                    f"{item['score_effectiveness']:.6f}",
                    f"{item['score_hand']:.6f}",
                    f"{item['score_target']:.6f}",
                    f"{item['fp_penalty']:.6f}",
                ]
            )
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
