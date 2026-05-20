#!/usr/bin/env python3
"""Generate a temporal-prior submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.temporal_prior import TemporalPriorConfig, make_temporal_prior_submission
from rascar_boxing.validation import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("submissions/temporal_prior.csv"))
    parser.add_argument("--group-cols", default="dataset_type")
    parser.add_argument("--count-mode", choices=["count", "rate"], default="count")
    parser.add_argument("--spacing", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--fighter-mode", choices=["blue", "red", "alternate"], default="blue")
    parser.add_argument("--count-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TemporalPriorConfig(
        group_cols=tuple(col.strip() for col in args.group_cols.split(",") if col.strip()),
        count_mode=args.count_mode,
        spacing=args.spacing,
        fighter_mode=args.fighter_mode,
        count_multiplier=args.count_multiplier,
    )
    make_temporal_prior_submission(args.data_root, args.output, config)

    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Wrote {args.output}")
    print(f"config={config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

