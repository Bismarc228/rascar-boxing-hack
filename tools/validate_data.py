#!/usr/bin/env python3
"""Validate local dataset files and an optional submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.validation import validate_dataset, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--check-video-props", action="store_true")
    parser.add_argument("--no-strict-id-metadata", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_dataset(args.data_root, check_video_props=args.check_video_props)

    if args.submission:
        errors.extend(
            validate_submission(
                args.submission,
                args.data_root / "sample_submission.csv",
                strict_id_metadata=not args.no_strict_id_metadata,
            )
        )

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
