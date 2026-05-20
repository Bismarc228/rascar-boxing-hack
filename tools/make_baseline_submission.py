#!/usr/bin/env python3
"""Generate a first valid submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.baseline import make_submission, most_common_attributes
from rascar_boxing.io import read_csv_rows
from rascar_boxing.validation import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--strategy",
        choices=["sample", "all_false", "train_priors"],
        default="sample",
        help="sample copies the sample submission; train_priors keeps sample frames but uses train label priors.",
    )
    parser.add_argument(
        "--clear-mode",
        choices=["strategy", "sample", "all_true", "all_false"],
        default="strategy",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    make_submission(args.data_root, args.output, strategy=args.strategy, clear_mode=args.clear_mode)
    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1

    train_rows = read_csv_rows(args.data_root / "train/punches.csv")
    priors = most_common_attributes(train_rows, clear_only=True)
    print(f"Wrote {args.output}")
    print(f"strategy={args.strategy}, clear_mode={args.clear_mode}")
    print(f"train clear-label priors={priors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
