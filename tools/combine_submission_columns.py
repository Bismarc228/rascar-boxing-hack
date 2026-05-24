#!/usr/bin/env python3
"""Combine selected columns from multiple submission-like CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Column source as column=path. Repeat for multiple columns.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_rows = read_csv_rows(args.base)
    id_order = [row["id"] for row in base_rows]
    output_rows = [{col: row.get(col, "") for col in SUBMISSION_COLUMNS} for row in base_rows]
    changed_by_column: dict[str, int] = {}
    for column, path in parse_sources(args.source):
        if column not in SUBMISSION_COLUMNS:
            raise ValueError(f"unsupported submission column: {column}")
        source_rows = read_csv_rows(path)
        if [row["id"] for row in source_rows] != id_order:
            raise RuntimeError(f"source does not align by id: {path}")
        changed = 0
        for output, base, source in zip(output_rows, base_rows, source_rows):
            output[column] = source.get(column, "")
            if output[column] != base.get(column, ""):
                changed += 1
        changed_by_column[column] = changed
    write_csv_rows(args.output, output_rows, SUBMISSION_COLUMNS)
    print(
        f"base={args.base} output={args.output} "
        f"changed_by_column={format_counts(changed_by_column)}",
        flush=True,
    )
    return 0


def parse_sources(specs: list[str]) -> list[tuple[str, Path]]:
    output = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--source must be column=path, got {spec!r}")
        column, path = spec.split("=", 1)
        output.append((column.strip(), Path(path)))
    return output


def format_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
