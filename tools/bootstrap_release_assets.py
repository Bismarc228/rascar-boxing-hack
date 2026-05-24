#!/usr/bin/env python3
"""Download external model assets and print cache rebuild commands for releases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("release_artifacts/release_artifacts_manifest.tsv")
VAL_KEYS = "agn_003,agn_004,agn_010,agn_023,agn_024,agn_025,agn_056,agn_057,agn_058,agn_069,agn_070,agn_071,agn_072"

HF_TARGETS = {
    "data/processed/model_zoo/hf/google__vit-base-patch16-224-in21k": (
        "google/vit-base-patch16-224-in21k",
        "b4569560a39a0f1af58e3ddaf17facf20ab919b0",
    ),
    "data/processed/model_zoo/hf/openai__clip-vit-base-patch16": (
        "openai/clip-vit-base-patch16",
        "57c216476eefef5ab752ec549e440a49ae4ae5f3",
    ),
    "data/processed/model_zoo/hf/MCG-NJU__videomae-base-finetuned-kinetics": (
        "MCG-NJU/videomae-base-finetuned-kinetics",
        "488eb9a0565f257b32866000305c8178965eb9f6",
    ),
    "data/processed/model_zoo/hf/facebook__dino-vitb16": (
        "facebook/dino-vitb16",
        "f205d5d8e640a89a2b8ef0369670dfc37cc07fc2",
    ),
}

TIMM_MODELS = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch16_224.dino",
]

TRACK_COMMANDS = {
    "shared": [
        [
            ".venv/bin/python",
            "tools/run_pose_batch.py",
            "--data-root",
            "data/raw",
            "--videos-csv",
            "data/raw/train/videos.csv",
            "--video-keys",
            VAL_KEYS,
            "--output-dir",
            "data/processed/pose_tracks/val_yolo26l_conf035",
            "--model",
            "yolo26l-pose.pt",
            "--conf",
            "0.35",
            "--cuda-visible-devices",
            "1",
            "--jobs",
            "1",
            "--no-tqdm",
        ],
        [
            ".venv/bin/python",
            "tools/run_pose_batch.py",
            "--data-root",
            "data/raw",
            "--videos-csv",
            "data/raw/test/videos.csv",
            "--output-dir",
            "data/processed/pose_tracks/test_yolo26l_conf035",
            "--model",
            "yolo26l-pose.pt",
            "--conf",
            "0.35",
            "--cuda-visible-devices",
            "1",
            "--jobs",
            "1",
            "--no-tqdm",
        ],
    ],
    "rc1": [
        [
            ".venv/bin/python",
            "tools/run_pose_batch.py",
            "--data-root",
            "data/raw",
            "--videos-csv",
            "data/raw/train/videos.csv",
            "--output-dir",
            "data/processed/pose_tracks/train_yolo11s_conf035",
            "--model",
            "models/yolo11s-pose.pt",
            "--conf",
            "0.35",
            "--cuda-visible-devices",
            "1",
            "--jobs",
            "1",
            "--no-tqdm",
        ],
    ],
    "rc2": [
        [
            ".venv/bin/python",
            "tools/run_pose_batch.py",
            "--data-root",
            "data/raw",
            "--videos-csv",
            "data/raw/train/videos.csv",
            "--video-keys",
            VAL_KEYS,
            "--output-dir",
            "data/processed/pose_tracks/val_yolo26x_conf035",
            "--model",
            "yolo26x-pose.pt",
            "--conf",
            "0.35",
            "--cuda-visible-devices",
            "1",
            "--jobs",
            "1",
            "--no-tqdm",
        ],
        [
            ".venv/bin/python",
            "tools/run_pose_batch.py",
            "--data-root",
            "data/raw",
            "--videos-csv",
            "data/raw/test/videos.csv",
            "--output-dir",
            "data/processed/pose_tracks/test_yolo26x_conf035",
            "--model",
            "yolo26x-pose.pt",
            "--conf",
            "0.35",
            "--cuda-visible-devices",
            "1",
            "--jobs",
            "1",
            "--no-tqdm",
        ],
        [
            ".venv/bin/python",
            "tools/run_pose_batch.py",
            "--data-root",
            "data/raw",
            "--videos-csv",
            "data/raw/train/videos.csv",
            "--video-keys",
            VAL_KEYS,
            "--output-dir",
            "data/processed/pose_tracks/val_yolo11s_conf035",
            "--model",
            "models/yolo11s-pose.pt",
            "--conf",
            "0.35",
            "--cuda-visible-devices",
            "1",
            "--jobs",
            "1",
            "--no-tqdm",
        ],
        [
            ".venv/bin/python",
            "tools/run_pose_batch.py",
            "--data-root",
            "data/raw",
            "--videos-csv",
            "data/raw/test/videos.csv",
            "--output-dir",
            "data/processed/pose_tracks/test_yolo11s_conf035",
            "--model",
            "models/yolo11s-pose.pt",
            "--conf",
            "0.35",
            "--cuda-visible-devices",
            "1",
            "--jobs",
            "1",
            "--no-tqdm",
        ],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-models")
    download.add_argument("--skip-yolo", action="store_true")
    download.add_argument("--skip-hf", action="store_true")
    download.add_argument("--skip-timm", action="store_true")
    download.add_argument("--ultralytics-release", default="v8.4.0")
    download.add_argument("--verify", action="store_true")

    tracks = subparsers.add_parser("print-track-commands")
    tracks.add_argument("--release", choices=["all", "rc1", "rc2"], default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "download-models":
        if not args.skip_yolo:
            download_yolo(args.ultralytics_release)
        if not args.skip_hf:
            download_hf()
        if not args.skip_timm:
            download_timm()
        if args.verify:
            subprocess.run(
                [
                    sys.executable,
                    "tools/verify_release_artifacts.py",
                    "--stage",
                    "external",
                    "--strict",
                ],
                cwd=ROOT,
                check=True,
            )
        return 0
    if args.command == "print-track-commands":
        print_track_commands(args.release)
        return 0
    raise AssertionError(args.command)


def download_yolo(release: str) -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    for row in manifest_rows():
        if row["stage"] != "external" or row["kind"] != "yolo_weight":
            continue
        path = ROOT / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"yolo_asset {row['path']} release={release}", flush=True)
        attempt_download_asset(path, release=release)


def download_hf() -> None:
    from huggingface_hub import snapshot_download

    for target, (repo_id, revision) in HF_TARGETS.items():
        filenames = sorted(
            Path(row["path"]).name
            for row in manifest_rows()
            if row["stage"] == "external" and row["kind"] == "hf_model" and row["path"].startswith(target + "/")
        )
        local_dir = ROOT / target
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"hf_snapshot {repo_id}@{revision} -> {target}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=filenames,
        )


def download_timm() -> None:
    import timm

    for model_name in TIMM_MODELS:
        print(f"timm_model {model_name}", flush=True)
        timm.create_model(model_name, pretrained=True).eval()


def print_track_commands(release: str) -> None:
    keys = ["shared"]
    if release in {"all", "rc1"}:
        keys.append("rc1")
    if release in {"all", "rc2"}:
        keys.append("rc2")
    for key in keys:
        for command in TRACK_COMMANDS[key]:
            print(shell_join(command))


def manifest_rows() -> list[dict[str, str]]:
    with (ROOT / MANIFEST).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def shell_join(command: list[str]) -> str:
    return " ".join(shell_quote(part) for part in command)


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "._/:=,+-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
