#!/usr/bin/env python3
"""Replace selected test-video rows in one submission with rows from another."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.validation import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--video-keys", required=True, help="Comma-separated video keys to replace")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keys = {value for value in args.video_keys.split(",") if value}
    if not keys:
        print("--video-keys must contain at least one key", file=sys.stderr)
        return 2

    base_rows = read_csv_rows(args.base)
    override_rows = read_csv_rows(args.override)
    override_by_id = {row["id"]: row for row in override_rows}

    output = []
    replaced = 0
    for row in base_rows:
        if row["video_key"] in keys:
            output.append(dict(override_by_id[row["id"]]))
            replaced += 1
        else:
            output.append(dict(row))

    write_csv_rows(args.output, output, SUBMISSION_COLUMNS)
    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        print("Generated submission is invalid:")
        for error in errors:
            print(f"  - {error}")
        return 1

    clear_by_video: dict[str, int] = {}
    for row in output:
        if row["clear"] == "true":
            clear_by_video[row["video_key"]] = clear_by_video.get(row["video_key"], 0) + 1

    print(f"Wrote {args.output}")
    print(f"replaced_rows={replaced} video_keys={','.join(sorted(keys))}")
    print("selected=" + ",".join(f"{key}:{clear_by_video[key]}" for key in sorted(clear_by_video)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
