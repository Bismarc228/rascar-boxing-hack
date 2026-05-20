#!/usr/bin/env python3
"""Evaluate pose-heuristic variants on cached validation tracks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, predict_for_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument(
        "--val-keys",
        type=Path,
        default=Path("data/processed/splits/fight_seed42/val_video_keys.txt"),
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-scores", default="0.04,0.06,0.08,0.10,0.12,0.16,0.20")
    parser.add_argument("--nms-frames", default="8,12,16,24")
    parser.add_argument("--frame-offsets", default="-6,-3,0,3,6")
    parser.add_argument("--count-modes", default="dataset_count,dataset_rate,threshold")
    parser.add_argument("--multipliers", default="0.4,0.6,0.8,1.0,1.2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    punches = read_csv_rows(args.data_root / "train/punches.csv")
    val_keys = set(args.val_keys.read_text(encoding="utf-8").split())
    train_videos = [row for row in videos if row["video_key"] not in val_keys]
    val_videos = [row for row in videos if row["video_key"] in val_keys]
    val_gt = [row for row in punches if row["video_key"] in val_keys]

    missing = [row["video_key"] for row in val_videos if not (args.tracks_dir / f"{row['video_key']}.jsonl").exists()]
    if missing:
        print(f"Missing tracks for {len(missing)} videos: {','.join(missing)}")
        return 2

    min_scores = [float(value) for value in args.min_scores.split(",") if value]
    nms_values = [int(value) for value in args.nms_frames.split(",") if value]
    frame_offsets = [int(value) for value in args.frame_offsets.split(",") if value]
    count_modes = [value for value in args.count_modes.split(",") if value]
    multipliers = [float(value) for value in args.multipliers.split(",") if value]

    results = []
    for count_mode in count_modes:
        for multiplier in multipliers:
            if count_mode == "threshold" and multiplier != multipliers[0]:
                continue
            for min_score in min_scores:
                for nms_frames in nms_values:
                    for frame_offset in frame_offsets:
                        config = PoseHeuristicConfig(
                            min_score=min_score,
                            nms_frames=nms_frames,
                            count_mode=count_mode,
                            count_multiplier=multiplier,
                            frame_offset=frame_offset,
                        )
                        pred = predict_for_videos(
                            val_videos,
                            args.tracks_dir,
                            config,
                            train_videos=train_videos,
                            train_punches=[row for row in punches if row["video_key"] not in val_keys],
                        )
                        score = score_predictions(val_gt, pred)["macro_score"]
                        results.append((score, len(pred), config))

    print("score,n_pred,count_mode,count_multiplier,min_score,nms_frames,frame_offset")
    for score, n_pred, config in sorted(results, key=lambda item: item[0], reverse=True)[: args.top_k]:
        print(
            f"{score:.6f},{n_pred},{config.count_mode},{config.count_multiplier},"
            f"{config.min_score},{config.nms_frames},{config.frame_offset}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
