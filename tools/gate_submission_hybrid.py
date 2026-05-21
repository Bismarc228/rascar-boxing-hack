#!/usr/bin/env python3
"""Gate per-video submission replacements by local frame/count consistency.

This is a submit-budget guardrail: compare a public-safe base submission with a
candidate override submission, then optionally write a hybrid that replaces
only videos whose selected frames stay locally close to the base.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.validation import validate_submission


@dataclass(frozen=True)
class VideoGate:
    video_key: str
    base_count: int
    override_count: int
    count_delta: int
    base_within_15: float
    base_within_30: float
    base_within_90: float
    override_within_15: float
    override_within_30: float
    override_within_90: float
    base_median_gap: float
    override_median_gap: float
    pass_gate: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--row-mode",
        choices=["sample", "video_rows"],
        default="sample",
        help="sample replaces rows by fixed Kaggle ids; video_rows replaces whole video row groups",
    )
    parser.add_argument("--video-keys", help="Optional comma-separated subset to consider")
    parser.add_argument("--max-count-delta", type=int, default=20)
    parser.add_argument("--min-base-within-15", type=float, default=0.80)
    parser.add_argument("--min-override-within-15", type=float, default=0.80)
    parser.add_argument("--max-median-gap", type=float, default=15.0)
    parser.add_argument("--force-include", default="", help="Comma-separated keys to include if present")
    parser.add_argument("--force-exclude", default="", help="Comma-separated keys to exclude even if gated")
    return parser.parse_args()


def clear_rows_by_video(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["clear"] == "true":
            by_video[row["video_key"]].append(row)
    for key in by_video:
        by_video[key].sort(key=lambda item: int(item["frame"]))
    return dict(by_video)


def nearest_gaps(source_frames: list[int], target_frames: list[int]) -> list[int]:
    if not source_frames:
        return []
    if not target_frames:
        return [10**9 for _ in source_frames]
    out: list[int] = []
    for frame in source_frames:
        pos = bisect_left(target_frames, frame)
        best = abs(frame - target_frames[min(pos, len(target_frames) - 1)])
        if pos > 0:
            best = min(best, abs(frame - target_frames[pos - 1]))
        out.append(best)
    return out


def rate_leq(values: list[int], threshold: int) -> float:
    if not values:
        return 1.0
    return sum(value <= threshold for value in values) / len(values)


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    items = sorted(values)
    mid = len(items) // 2
    if len(items) % 2:
        return float(items[mid])
    return float(items[mid - 1] + items[mid]) / 2.0


def split_keys(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def evaluate_video(
    video_key: str,
    base_rows: list[dict[str, str]],
    override_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> VideoGate:
    base_frames = [int(row["frame"]) for row in base_rows]
    override_frames = [int(row["frame"]) for row in override_rows]
    base_gaps = nearest_gaps(base_frames, override_frames)
    override_gaps = nearest_gaps(override_frames, base_frames)
    count_delta = len(override_frames) - len(base_frames)
    pass_gate = (
        abs(count_delta) <= args.max_count_delta
        and rate_leq(base_gaps, 15) >= args.min_base_within_15
        and rate_leq(override_gaps, 15) >= args.min_override_within_15
        and median(base_gaps) <= args.max_median_gap
        and median(override_gaps) <= args.max_median_gap
    )
    return VideoGate(
        video_key=video_key,
        base_count=len(base_frames),
        override_count=len(override_frames),
        count_delta=count_delta,
        base_within_15=rate_leq(base_gaps, 15),
        base_within_30=rate_leq(base_gaps, 30),
        base_within_90=rate_leq(base_gaps, 90),
        override_within_15=rate_leq(override_gaps, 15),
        override_within_30=rate_leq(override_gaps, 30),
        override_within_90=rate_leq(override_gaps, 90),
        base_median_gap=median(base_gaps),
        override_median_gap=median(override_gaps),
        pass_gate=pass_gate,
    )


def print_gates(gates: list[VideoGate]) -> None:
    print(
        "video_key,base_count,override_count,delta,"
        "base_le15,base_le30,base_le90,override_le15,override_le30,override_le90,"
        "base_med_gap,override_med_gap,pass"
    )
    for gate in gates:
        print(
            f"{gate.video_key},{gate.base_count},{gate.override_count},{gate.count_delta},"
            f"{gate.base_within_15:.4f},{gate.base_within_30:.4f},{gate.base_within_90:.4f},"
            f"{gate.override_within_15:.4f},{gate.override_within_30:.4f},{gate.override_within_90:.4f},"
            f"{gate.base_median_gap:.1f},{gate.override_median_gap:.1f},{int(gate.pass_gate)}"
        )


def write_hybrid(
    output_path: Path,
    data_root: Path,
    base_rows: list[dict[str, str]],
    override_rows: list[dict[str, str]],
    replace_keys: set[str],
    row_mode: str,
) -> None:
    output: list[dict[str, str]] = []
    if row_mode == "sample":
        override_by_id = {row["id"]: row for row in override_rows}
        for row in base_rows:
            if row["video_key"] in replace_keys:
                output.append(dict(override_by_id[row["id"]]))
            else:
                output.append(dict(row))
    elif row_mode == "video_rows":
        for row in base_rows:
            if row["video_key"] not in replace_keys:
                output.append(dict(row))
        for row in override_rows:
            if row["video_key"] in replace_keys:
                output.append(dict(row))
        output.sort(key=lambda row: (row["video_key"], int(row["frame"]), row["fighter"], row["hand"]))
        for index, row in enumerate(output, start=1):
            row["id"] = str(index)
    else:
        raise ValueError(f"Unknown row mode: {row_mode}")

    write_csv_rows(output_path, output, SUBMISSION_COLUMNS)
    if row_mode == "sample":
        errors = validate_submission(output_path, data_root / "sample_submission.csv")
        if errors:
            for error in errors:
                print(f"validation_error={error}", file=sys.stderr)
            raise SystemExit(1)
    selected = Counter(row["video_key"] for row in output if row["clear"] == "true")
    print(f"wrote={output_path}")
    print(f"row_mode={row_mode}")
    print(f"replace_keys={','.join(sorted(replace_keys))}")
    print("selected=" + ",".join(f"{key}:{selected[key]}" for key in sorted(selected)))


def main() -> int:
    args = parse_args()
    base_rows = read_csv_rows(args.base)
    override_rows = read_csv_rows(args.override)
    base_by_video = clear_rows_by_video(base_rows)
    override_by_video = clear_rows_by_video(override_rows)
    all_keys = set(base_by_video) | set(override_by_video)
    requested = split_keys(args.video_keys)
    if requested:
        all_keys &= requested
    force_include = split_keys(args.force_include)
    force_exclude = split_keys(args.force_exclude)

    gates = [
        evaluate_video(key, base_by_video.get(key, []), override_by_video.get(key, []), args)
        for key in sorted(all_keys)
    ]
    print_gates(gates)
    replace_keys = {gate.video_key for gate in gates if gate.pass_gate}
    replace_keys |= force_include
    replace_keys -= force_exclude
    replace_keys &= all_keys
    print(f"auto_replace_keys={','.join(sorted(gate.video_key for gate in gates if gate.pass_gate))}")
    print(f"final_replace_keys={','.join(sorted(replace_keys))}")

    if args.output:
        write_hybrid(args.output, args.data_root, base_rows, override_rows, replace_keys, args.row_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
