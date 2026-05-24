#!/usr/bin/env python3
"""Extract RGB crop features for a reconstructed pose-candidate pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_rows, write_csv_rows
from tools.evaluate_rgb_contact_clip import load_or_extract_features
from tools.make_rgb_fixed_row_attribute_submission import build_candidate_pool, candidate_feature_rows


INDEX_COLUMNS = [
    "feature_index",
    "id",
    "video_key",
    "frame",
    "fighter",
    "hand",
    "target",
    "punch_type",
    "effectiveness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--index-output", type=Path)
    parser.add_argument("--model-name", default="vit_small_patch16_224.dino")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--crop-modes", default="union")
    parser.add_argument("--crop-expand", type=float, default=0.16)
    parser.add_argument("--decode-mode", choices=["sequential", "seek"], default="sequential")
    parser.add_argument("--video-decoder", choices=["opencv", "ffmpeg_cuda"], default="opencv")
    parser.add_argument("--gpu-preprocess", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=1800)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    video_by_key = {row["video_key"]: row for row in read_csv_rows(args.data_root / "train/videos.csv")}
    missing = sorted(set(keys) - set(video_by_key))
    if missing:
        raise RuntimeError(f"Prediction keys are not train videos: {missing}")

    candidates_by_key = build_candidate_pool(args, keys, args.tracks_dir)
    candidate_rows = candidate_feature_rows(candidates_by_key, video_by_key)
    cache_args = SimpleNamespace(
        data_root=args.data_root,
        tracks_dir=args.tracks_dir,
        feature_cache=args.feature_cache,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        crop_modes=args.crop_modes,
        crop_expand=args.crop_expand,
        decode_mode=args.decode_mode,
        video_decoder=args.video_decoder,
        gpu_preprocess=args.gpu_preprocess,
        cpu_threads=args.cpu_threads,
        quiet=args.quiet,
    )
    features = load_or_extract_features(cache_args, candidate_rows, video_by_key)
    if len(features) != len(candidate_rows):
        raise RuntimeError(f"features={len(features)} candidate_rows={len(candidate_rows)}")

    index_path = args.index_output or args.feature_cache.with_name(args.feature_cache.stem + "_index.csv")
    index_rows = []
    for index, row in enumerate(candidate_rows):
        item = {column: row.get(column, "") for column in INDEX_COLUMNS}
        item["feature_index"] = str(index)
        index_rows.append(item)
    write_csv_rows(index_path, index_rows, INDEX_COLUMNS)
    print(
        f"features={features.shape} candidates={len(candidate_rows)} "
        f"model={args.model_name} cache={args.feature_cache} index={index_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
