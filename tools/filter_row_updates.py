#!/usr/bin/env python3
"""Copy selected row updates from a candidate CSV under simple value filters."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.io import read_csv_header, read_csv_rows, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--copy-columns",
        required=True,
        help="Comma-separated columns to copy from candidate when filters pass.",
    )
    parser.add_argument(
        "--base-filter",
        action="append",
        default=[],
        help="Filter as column=value1,value2,... applied to the base row.",
    )
    parser.add_argument(
        "--candidate-filter",
        action="append",
        default=[],
        help="Filter as column=value1,value2,... applied to the candidate row.",
    )
    parser.add_argument(
        "--transition-filter",
        action="append",
        default=[],
        help=(
            "Filter as column:before->after,before2->after2,... applied to "
            "base/candidate transitions."
        ),
    )
    parser.add_argument(
        "--fail-if-empty",
        action="store_true",
        help="Exit non-zero when no row update matches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    header = read_csv_header(args.base)
    candidate_header = read_csv_header(args.candidate)
    copy_columns = parse_columns(args.copy_columns)
    missing = [column for column in copy_columns if column not in header]
    if missing:
        raise SystemExit(f"base missing copy columns: {','.join(missing)}")
    missing = [column for column in copy_columns if column not in candidate_header]
    if missing:
        raise SystemExit(f"candidate missing copy columns: {','.join(missing)}")

    base_filters = parse_filters(args.base_filter)
    candidate_filters = parse_filters(args.candidate_filter)
    transition_filters = parse_transition_filters(args.transition_filter)
    base_rows = read_csv_rows(args.base)
    candidate_rows = read_csv_rows(args.candidate)
    ensure_aligned(base_rows, candidate_rows)

    output_rows: list[dict[str, str]] = []
    selected = 0
    changed_total = 0
    changed_by_column: Counter[str] = Counter()
    selected_by_video: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for base, candidate in zip(base_rows, candidate_rows):
        row = dict(base)
        has_change = any(base[column] != candidate[column] for column in copy_columns)
        if has_change:
            changed_total += 1
        if (
            has_change
            and matches(base, base_filters)
            and matches(candidate, candidate_filters)
            and matches_transitions(base, candidate, transition_filters)
        ):
            selected += 1
            selected_by_video[base.get("video_key", "")] += 1
            for column in copy_columns:
                before = base[column]
                after = candidate[column]
                row[column] = after
                if before != after:
                    changed_by_column[column] += 1
                    transitions[f"{column}:{before}->{after}"] += 1
        output_rows.append(row)

    if args.fail_if_empty and selected == 0:
        raise SystemExit("no row updates matched filters")
    write_csv_rows(args.output, output_rows, header)

    print(f"base={args.base}")
    print(f"candidate={args.candidate}")
    print(f"output={args.output}")
    print(f"changed_total={changed_total}")
    print(f"selected={selected}")
    print(f"changed_by_column={format_counts(changed_by_column)}")
    print(f"selected_by_video={format_counts(selected_by_video)}")
    print(f"transitions={format_counts(transitions)}")
    return 0


def parse_columns(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise ValueError("--copy-columns is empty")
    return columns


def parse_filters(specs: list[str]) -> list[tuple[str, set[str]]]:
    filters: list[tuple[str, set[str]]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"filter must be column=value1,value2,..., got {spec!r}")
        column, values = spec.split("=", 1)
        allowed = {item.strip() for item in values.split(",") if item.strip()}
        if not column.strip() or not allowed:
            raise ValueError(f"invalid filter: {spec!r}")
        filters.append((column.strip(), allowed))
    return filters


def parse_transition_filters(specs: list[str]) -> list[tuple[str, set[tuple[str, str]]]]:
    filters: list[tuple[str, set[tuple[str, str]]]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"transition filter must be column:before->after,..., got {spec!r}")
        column, transitions_text = spec.split(":", 1)
        transitions: set[tuple[str, str]] = set()
        for item in transitions_text.split(","):
            item = item.strip()
            if not item:
                continue
            if "->" not in item:
                raise ValueError(f"invalid transition item {item!r} in {spec!r}")
            before, after = item.split("->", 1)
            before = before.strip()
            after = after.strip()
            if not before or not after:
                raise ValueError(f"invalid transition item {item!r} in {spec!r}")
            transitions.add((before, after))
        if not column.strip() or not transitions:
            raise ValueError(f"invalid transition filter: {spec!r}")
        filters.append((column.strip(), transitions))
    return filters


def ensure_aligned(base_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]) -> None:
    if len(base_rows) != len(candidate_rows):
        raise SystemExit(f"row count mismatch: base={len(base_rows)} candidate={len(candidate_rows)}")
    preferred_keys = ["id", "video_key", "frame"]
    keys = [key for key in preferred_keys if base_rows and key in base_rows[0] and key in candidate_rows[0]]
    if not keys:
        return
    base_ids = [tuple(row.get(key, "") for key in keys) for row in base_rows]
    candidate_ids = [tuple(row.get(key, "") for key in keys) for row in candidate_rows]
    if base_ids != candidate_ids:
        raise SystemExit(f"base and candidate are not aligned by {','.join(keys)}")


def matches(row: dict[str, str], filters: list[tuple[str, set[str]]]) -> bool:
    return all(row.get(column, "") in allowed for column, allowed in filters)


def matches_transitions(
    base: dict[str, str],
    candidate: dict[str, str],
    filters: list[tuple[str, set[tuple[str, str]]]],
) -> bool:
    return all((base.get(column, ""), candidate.get(column, "")) in allowed for column, allowed in filters)


def format_counts(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


if __name__ == "__main__":
    raise SystemExit(main())
