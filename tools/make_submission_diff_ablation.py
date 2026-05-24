#!/usr/bin/env python3
"""Build fixed-id submission ablations from a base and override CSV."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_header, read_csv_rows, write_csv_rows
from rascar_boxing.validation import validate_submission


def parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", help="Comma-separated changed ids to include")
    parser.add_argument("--exclude-ids", help="Comma-separated changed ids to exclude")
    parser.add_argument("--video-keys", help="Comma-separated video_key values to include")
    parser.add_argument("--exclude-video-keys", help="Comma-separated video_key values to exclude")
    parser.add_argument("--effectiveness", help="Comma-separated effectiveness values to include")
    parser.add_argument("--exclude-effectiveness", help="Comma-separated effectiveness values to exclude")
    parser.add_argument("--fail-if-empty", action="store_true")
    return parser.parse_args()


def ensure_submission_shape(path: Path, rows: list[dict[str, str]]) -> None:
    header = read_csv_header(path)
    if header != SUBMISSION_COLUMNS:
        raise SystemExit(f"{path}: expected submission columns {SUBMISSION_COLUMNS}, got {header}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate ids")


def changed_ids(
    base_rows: list[dict[str, str]],
    override_rows: list[dict[str, str]],
) -> list[str]:
    base_ids = [row["id"] for row in base_rows]
    override_ids = [row["id"] for row in override_rows]
    if base_ids != override_ids:
        raise SystemExit("base and override must have the same ids in the same order")
    changed: list[str] = []
    for base_row, override_row in zip(base_rows, override_rows):
        if any(base_row[col] != override_row[col] for col in SUBMISSION_COLUMNS):
            changed.append(base_row["id"])
    return changed


def row_passes_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    include_ids = parse_csv_set(args.ids)
    exclude_ids = parse_csv_set(args.exclude_ids)
    include_videos = parse_csv_set(args.video_keys)
    exclude_videos = parse_csv_set(args.exclude_video_keys)
    include_effectiveness = parse_csv_set(args.effectiveness)
    exclude_effectiveness = parse_csv_set(args.exclude_effectiveness)

    if include_ids and row["id"] not in include_ids:
        return False
    if row["id"] in exclude_ids:
        return False
    if include_videos and row["video_key"] not in include_videos:
        return False
    if row["video_key"] in exclude_videos:
        return False
    if include_effectiveness and row["effectiveness"] not in include_effectiveness:
        return False
    if row["effectiveness"] in exclude_effectiveness:
        return False
    return True


def print_summary(
    output: Path,
    base_rows: list[dict[str, str]],
    override_by_id: dict[str, dict[str, str]],
    selected_ids: list[str],
    total_changed: int,
) -> None:
    selected_rows = [override_by_id[row_id] for row_id in selected_ids]
    base_clear = sum(row["clear"] == "true" for row in base_rows)
    selected_clear_delta = sum(
        (override_by_id[row["id"]]["clear"] == "true") - (row["clear"] == "true")
        for row in base_rows
        if row["id"] in selected_ids
    )
    output_clear = base_clear + selected_clear_delta
    by_video = Counter(row["video_key"] for row in selected_rows)
    by_effectiveness = Counter(row["effectiveness"] for row in selected_rows)
    print(f"wrote={output}")
    print(f"changed_total={total_changed}")
    print(f"selected={len(selected_ids)}")
    print(f"base_clear={base_clear}")
    print(f"output_clear={output_clear}")
    print(f"selected_clear_delta={selected_clear_delta:+d}")
    print("selected_ids=" + ",".join(selected_ids))
    print("by_video=" + ",".join(f"{key}:{by_video[key]}" for key in sorted(by_video)))
    print(
        "by_effectiveness="
        + ",".join(f"{key}:{by_effectiveness[key]}" for key in sorted(by_effectiveness))
    )


def main() -> int:
    args = parse_args()
    base_rows = read_csv_rows(args.base)
    override_rows = read_csv_rows(args.override)
    ensure_submission_shape(args.base, base_rows)
    ensure_submission_shape(args.override, override_rows)

    override_by_id = {row["id"]: row for row in override_rows}
    all_changed_ids = changed_ids(base_rows, override_rows)
    selected_ids = [
        row_id
        for row_id in all_changed_ids
        if row_passes_filters(override_by_id[row_id], args)
    ]
    if args.fail_if_empty and not selected_ids:
        raise SystemExit("no changed rows matched filters")

    output_rows = [
        dict(override_by_id[row["id"]]) if row["id"] in selected_ids else dict(row)
        for row in base_rows
    ]
    write_csv_rows(args.output, output_rows, SUBMISSION_COLUMNS)

    errors = validate_submission(args.output, args.data_root / "sample_submission.csv")
    if errors:
        for error in errors:
            print(f"validation_error={error}", file=sys.stderr)
        return 1

    print_summary(args.output, base_rows, override_by_id, selected_ids, len(all_changed_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
