#!/usr/bin/env python3
"""Run pose baseline for multiple videos with skip/resume-friendly outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--videos-csv", type=Path, default=Path("data/raw/test/videos.csv"))
    parser.add_argument("--video-keys")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pose_tracks"))
    parser.add_argument("--model", default="models/yolo11n-pose.pt")
    parser.add_argument("--device")
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices and args.device:
        raise SystemExit(
            "Do not combine --cuda-visible-devices with --device for Ultralytics in this repo. "
            "Use --cuda-visible-devices 1 and omit --device so physical GPU 1 remains selected."
        )
    rows = read_csv_rows(args.videos_csv)
    if args.video_keys:
        wanted = set(args.video_keys.split(","))
        rows = [row for row in rows if row["video_key"] in wanted]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, args.jobs)
    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(process_video, args, row): row["video_key"]
            for row in rows
        }
        for future in as_completed(futures):
            video_key = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((video_key, exc))
                print(f"failed {video_key}: {exc}", file=sys.stderr, flush=True)
    if failures:
        return 1
    return 0


def process_video(args: argparse.Namespace, row: dict[str, str]) -> None:
    video_key = row["video_key"]
    output = args.output_dir / f"{video_key}.jsonl"
    if output.exists() and not args.overwrite:
        print(f"skip existing {output}")
        return

    cmd = [
        sys.executable,
        str(ROOT / "tools/run_pose_baseline.py"),
        "--data-root",
        str(args.data_root),
        "--video-key",
        video_key,
        "--model",
        args.model,
        "--tracks-out",
        str(output),
        "--conf",
        str(args.conf),
        "--progress-every",
        "0",
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.max_frames is not None:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.no_tqdm:
        cmd += ["--no-tqdm"]
    print("run", " ".join(cmd), flush=True)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
