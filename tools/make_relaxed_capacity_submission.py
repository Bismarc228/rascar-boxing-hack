#!/usr/bin/env python3
"""Create a no-submit diagnostic CSV that reallocates spare ids across videos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import (
    PoseHeuristicConfig,
    PunchCandidate,
    apply_temporal_context,
    estimate_attrs,
    estimate_count,
    fit_attr_priors,
    score_pose_tracks,
    select_candidates,
)
from rascar_boxing.validation import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, default=Path("submission.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-video-keys", default="agn_037")
    parser.add_argument("--donor-video-key", default="agn_062")
    parser.add_argument("--max-extra-per-video", type=int, default=20)
    parser.add_argument("--min-distance-frames", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--nms-frames", type=int, default=10)
    parser.add_argument("--cross-nms-frames", type=int, default=4)
    parser.add_argument("--count-mode", default="root_rate")
    parser.add_argument("--count-multiplier", type=float, default=0.88)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--strict-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_path = args.data_root / "sample_submission.csv"
    train_videos = read_csv_rows(args.data_root / "train/videos.csv")
    train_punches = [
        row for row in read_csv_rows(args.data_root / "train/punches.csv") if row["clear"] == "true"
    ]
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    video_by_key = {row["video_key"]: row for row in test_videos}
    attr_priors = fit_attr_priors(train_punches)
    output_rows = [dict(row) for row in read_csv_rows(args.anchor)]
    requested_keys = [key for key in args.extra_video_keys.split(",") if key]
    config = PoseHeuristicConfig(
        min_score=args.threshold,
        nms_frames=args.nms_frames,
        nms_group_mode="fighter_hand",
        cross_nms_frames=args.cross_nms_frames,
        count_mode=args.count_mode,
        count_multiplier=args.count_multiplier,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )

    donor_indexes = [
        index
        for index, row in enumerate(output_rows)
        if row["video_key"] == args.donor_video_key and row["clear"] == "false"
    ]
    if not donor_indexes:
        raise RuntimeError(f"No clear=false donor rows found for {args.donor_video_key}")

    donor_cursor = 0
    inserted_by_key: dict[str, int] = {}
    for key in requested_keys:
        video = video_by_key.get(key)
        if video is None:
            raise KeyError(f"Unknown test video key: {key}")
        candidates = no_cap_candidates(args.tracks_dir, video, train_videos, train_punches, config)
        extras = select_extra_candidates(
            key,
            candidates,
            output_rows,
            args.max_extra_per_video,
            args.min_distance_frames,
        )
        for candidate in extras:
            if donor_cursor >= len(donor_indexes):
                raise RuntimeError("Ran out of donor filler rows")
            row_index = donor_indexes[donor_cursor]
            donor_cursor += 1
            attrs = estimate_attrs(candidate, attr_priors)
            output_rows[row_index].update(
                {
                    "video_id": video["video_id"],
                    "agn_index": video["agn_index"],
                    "video_key": video["video_key"],
                    "frame": str(max(0, min(int(video["frame_count"]) - 1, candidate.frame))),
                    "fighter": candidate.fighter,
                    "punch_type": attrs["punch_type"],
                    "hand": candidate.hand,
                    "target": candidate.target,
                    "effectiveness": attrs["effectiveness"],
                    "clear": "true",
                }
            )
        inserted_by_key[key] = len(extras)

    write_csv_rows(args.output, output_rows, SUBMISSION_COLUMNS)
    relaxed_errors = validate_submission(args.output, sample_path, strict_id_metadata=False)
    strict_errors = validate_submission(args.output, sample_path, strict_id_metadata=True)
    print(f"Wrote {args.output}")
    print(f"donor={args.donor_video_key} donor_rows_used={donor_cursor}")
    print("inserted=" + ",".join(f"{key}:{inserted_by_key[key]}" for key in requested_keys))
    print_counts("clear_counts", output_rows)
    print(f"relaxed_validation_errors={len(relaxed_errors)}")
    print(f"strict_validation_errors={len(strict_errors)}")
    if strict_errors:
        print("strict_validation_first_error=" + strict_errors[0])
    if relaxed_errors:
        for error in relaxed_errors[:20]:
            print(f"relaxed_error={error}")
        return 1
    if args.strict_check and strict_errors:
        return 1
    return 0


def no_cap_candidates(
    tracks_dir: Path,
    video: dict[str, str],
    train_videos: list[dict[str, str]],
    train_punches: list[dict[str, str]],
    config: PoseHeuristicConfig,
) -> list[PunchCandidate]:
    raw = score_pose_tracks(tracks_dir / f"{video['video_key']}.jsonl", PoseHeuristicConfig(min_score=0.0))
    candidates = apply_temporal_context(raw, config)
    count = estimate_count(config, video, train_videos, train_punches, [])
    return select_candidates(candidates, config, count)


def select_extra_candidates(
    video_key: str,
    candidates: list[PunchCandidate],
    anchor_rows: list[dict[str, str]],
    max_extra: int,
    min_distance: int,
) -> list[PunchCandidate]:
    existing_frames = [
        int(row["frame"])
        for row in anchor_rows
        if row["video_key"] == video_key and row["clear"] == "true"
    ]
    output = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(abs(candidate.frame - frame) <= min_distance for frame in existing_frames):
            continue
        if any(abs(candidate.frame - item.frame) <= min_distance for item in output):
            continue
        output.append(candidate)
        if len(output) >= max_extra:
            break
    return sorted(output, key=lambda item: item.frame)


def print_counts(label: str, rows: list[dict[str, str]]) -> None:
    counts = {}
    for row in rows:
        if row["clear"] == "true":
            counts[row["video_key"]] = counts.get(row["video_key"], 0) + 1
    print(label + "=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    raise SystemExit(main())
