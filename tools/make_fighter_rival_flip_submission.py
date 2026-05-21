#!/usr/bin/env python3
"""Apply a fixed opposite-fighter rival rule to an existing submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_fighter_rival_flip import (
    apply_rival_rule,
    index_candidates,
    rival_predicate_factory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument(
        "--match-mode",
        choices=["same_hand_target", "same_hand", "any"],
        default="same_hand",
    )
    parser.add_argument("--ratio", type=float, default=1.3)
    parser.add_argument("--min-rival-score", type=float, default=1.0)
    parser.add_argument(
        "--update-mode",
        choices=["fighter_only", "fighter_hand_target"],
        default="fighter_hand_target",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.input)
    clear_rows = [row for row in rows if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in clear_rows})
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    candidates_by_key = {
        key: index_candidates(
            apply_temporal_context(
                score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config),
                config,
            )
        )
        for key in keys
    }
    modified_clear, changed = apply_rival_rule(
        clear_rows,
        candidates_by_key,
        args.window,
        rival_predicate_factory(args.match_mode),
        args.ratio,
        args.min_rival_score,
        args.update_mode,
    )
    modified_by_id = {row["id"]: row for row in modified_clear}
    output = []
    for row in rows:
        if row.get("clear") == "true" and row["id"] in modified_by_id:
            output.append(modified_by_id[row["id"]])
        else:
            output.append({col: row.get(col, "") for col in SUBMISSION_COLUMNS})
    write_csv_rows(args.output, output, SUBMISSION_COLUMNS)
    print(
        f"changed={changed} clear_rows={len(clear_rows)} total_rows={len(rows)} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
