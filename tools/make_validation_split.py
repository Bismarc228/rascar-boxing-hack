#!/usr/bin/env python3
"""Create a deterministic fight-level train/validation split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.split import make_fight_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-csv", type=Path, default=Path("data/raw/train/videos.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/splits/fight_seed42"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_keys, val_keys = make_fight_split(
        args.videos_csv,
        args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Wrote split to {args.output_dir}")
    print(f"train videos: {len(train_keys)}")
    print(f"val videos: {len(val_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
