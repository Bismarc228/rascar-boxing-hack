#!/usr/bin/env python3
"""Materialize a video-level source switch from two row CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--video-keys", required=True, help="Comma-separated video keys to take from override")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {item.strip() for item in args.video_keys.split(",") if item.strip()}
    if not selected:
        raise SystemExit("--video-keys must contain at least one video key")

    base_rows = read_csv_rows(args.base)
    override_rows = read_csv_rows(args.override)
    available = {row["video_key"] for row in override_rows}
    missing = sorted(selected - available)
    if missing:
        raise SystemExit("override is missing selected video keys: " + ",".join(missing))

    rows = [dict(row) for row in base_rows if row["video_key"] not in selected]
    rows.extend(dict(row) for row in override_rows if row["video_key"] in selected)
    rows.sort(key=lambda row: (row["video_key"], as_int(row["frame"], "frame"), row["fighter"], row["hand"]))
    for index, row in enumerate(rows, start=1):
        row["id"] = str(index)

    write_csv_rows(args.output, rows, SUBMISSION_COLUMNS)
    print(f"wrote={args.output}")
    print(f"selected={','.join(sorted(selected))}")
    print(f"rows={len(rows)} clear={sum(row.get('clear') == 'true' for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
