#!/usr/bin/env python3
"""Audit fighter-label errors for a fixed prediction row source."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from tools.evaluate_pose_selection_variants import video_wins


FIGHTERS = {"red", "blue"}


@dataclass
class TrackSummary:
    track_id: int
    role_majority: str
    role_purity: float
    color_label: str
    color_margin: float
    red_mean: float
    blue_mean: float
    n_frames: int
    first_frame: int
    last_frame: int


@dataclass
class TrackInfo:
    frame_role_track: dict[tuple[int, str], int]
    frame_role_record: dict[tuple[int, str], dict[str, Any]]
    summaries: dict[int, TrackSummary]


@dataclass(frozen=True)
class Rule:
    name: str
    suggest: Callable[[dict[str, str], TrackInfo], str | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    if not keys:
        print("No clear prediction rows found")
        return 2

    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    missing_videos = sorted(set(keys) - set(video_by_key))
    if missing_videos:
        raise RuntimeError(f"Prediction keys are not train videos: {missing_videos}")

    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    track_info_by_key = {
        key: load_track_info(args.tracks_dir / f"{key}.jsonl")
        for key in keys
    }

    baseline = score_predictions(gt_rows, pred_rows)
    rows_by_video = group_by(pred_rows, "video_key")
    gt_by_video = group_by(gt_rows, "video_key")
    audit_rows = build_audit_rows(rows_by_video, gt_by_video, video_by_key, track_info_by_key)
    output_path = args.output or default_output_path(args.predictions)
    write_csv_rows(output_path, audit_rows, AUDIT_COLUMNS)

    oracle_rows = oracle_matched_fighter_rows(rows_by_video, gt_by_video)
    oracle = score_predictions(gt_rows, oracle_rows)
    print_score("baseline", baseline, len(pred_rows), 0)
    print_score("oracle_matched_fighter", oracle, len(oracle_rows), video_wins(oracle, baseline))
    print_identity_summary(audit_rows, baseline, video_by_key, args.top_k)
    print_rule_scores(pred_rows, gt_rows, track_info_by_key, baseline, args.top_k)
    print(f"wrote={output_path}")
    return 0


AUDIT_COLUMNS = [
    "video_key",
    "data_root",
    "fight_index",
    "round_number",
    "row_id",
    "pred_index",
    "match_type",
    "pred_frame",
    "gt_frame",
    "frame_error",
    "abs_frame_error",
    "time_error_s",
    "scorable",
    "pred_fighter",
    "gt_fighter",
    "fighter_correct",
    "pred_hand",
    "gt_hand",
    "hand_correct",
    "pred_target",
    "gt_target",
    "target_correct",
    "pred_punch_type",
    "gt_punch_type",
    "pred_effectiveness",
    "gt_effectiveness",
    "pred_track_id",
    "gt_role_track_id",
    "pred_gt_tracks_equal",
    "pred_role_majority",
    "pred_role_purity",
    "pred_color_label",
    "pred_color_margin",
    "pred_red_mean",
    "pred_blue_mean",
    "pred_frame_color_label",
    "pred_frame_color_margin",
    "pred_track_frames",
    "pred_track_first",
    "pred_track_last",
    "rule_track_role",
    "rule_track_color",
    "rule_frame_color",
    "rule_role_color_agree",
]


def load_track_info(path: Path) -> TrackInfo:
    if not path.exists():
        raise FileNotFoundError(path)

    frame_role_track: dict[tuple[int, str], int] = {}
    frame_role_record: dict[tuple[int, str], dict[str, Any]] = {}
    role_counts: dict[int, Counter[str]] = defaultdict(Counter)
    score_sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    frame_bounds: dict[int, list[int]] = {}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = int(record["frame"])
            for role, fighter in record.get("fighters", {}).items():
                if role not in FIGHTERS or "track_id" not in fighter:
                    continue
                track_id = int(fighter["track_id"])
                frame_role_track[(frame, role)] = track_id
                frame_role_record[(frame, role)] = fighter
                role_counts[track_id][role] += 1
                score_sums[track_id][0] += float(fighter.get("score_red", 0.0))
                score_sums[track_id][1] += float(fighter.get("score_blue", 0.0))
                score_sums[track_id][2] += 1.0
                if track_id not in frame_bounds:
                    frame_bounds[track_id] = [frame, frame]
                else:
                    frame_bounds[track_id][0] = min(frame_bounds[track_id][0], frame)
                    frame_bounds[track_id][1] = max(frame_bounds[track_id][1], frame)

    summaries = {}
    for track_id, counts in role_counts.items():
        role_majority, role_count = counts.most_common(1)[0]
        red_sum, blue_sum, n_seen = score_sums[track_id]
        n_frames = int(n_seen)
        red_mean = red_sum / max(1.0, n_seen)
        blue_mean = blue_sum / max(1.0, n_seen)
        first_frame, last_frame = frame_bounds[track_id]
        summaries[track_id] = TrackSummary(
            track_id=track_id,
            role_majority=role_majority,
            role_purity=role_count / max(1, n_frames),
            color_label="red" if red_mean >= blue_mean else "blue",
            color_margin=red_mean - blue_mean,
            red_mean=red_mean,
            blue_mean=blue_mean,
            n_frames=n_frames,
            first_frame=first_frame,
            last_frame=last_frame,
        )
    return TrackInfo(frame_role_track, frame_role_record, summaries)


def build_audit_rows(
    rows_by_video: dict[str, list[dict[str, str]]],
    gt_by_video: dict[str, list[dict[str, str]]],
    video_by_key: dict[str, dict[str, str]],
    track_info_by_key: dict[str, TrackInfo],
) -> list[dict[str, object]]:
    output = []
    for key in sorted(rows_by_video):
        pred_rows = rows_by_video[key]
        gt_rows = gt_by_video.get(key, [])
        match_by_pred = {match.pred_index: match.gt_index for match in match_events(gt_rows, pred_rows)}
        video = video_by_key[key]
        track_info = track_info_by_key[key]
        for pred_index, pred in enumerate(pred_rows):
            gt_index = match_by_pred.get(pred_index)
            gt = gt_rows[gt_index] if gt_index is not None else None
            output.append(
                audit_row(video, pred, pred_index, gt, track_info)
            )
    return output


def oracle_matched_fighter_rows(
    rows_by_video: dict[str, list[dict[str, str]]],
    gt_by_video: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    output = []
    for key in sorted(rows_by_video):
        pred_rows = rows_by_video[key]
        gt_rows = gt_by_video.get(key, [])
        fighter_by_pred = {
            match.pred_index: gt_rows[match.gt_index]["fighter"]
            for match in match_events(gt_rows, pred_rows)
        }
        for pred_index, row in enumerate(pred_rows):
            out = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
            if pred_index in fighter_by_pred:
                out["fighter"] = fighter_by_pred[pred_index]
            output.append(out)
    return output


def audit_row(
    video: dict[str, str],
    pred: dict[str, str],
    pred_index: int,
    gt: dict[str, str] | None,
    track_info: TrackInfo,
) -> dict[str, object]:
    frame = as_int(pred["frame"], "frame")
    pred_fighter = pred["fighter"]
    pred_track_id = track_info.frame_role_track.get((frame, pred_fighter))
    summary = track_info.summaries.get(pred_track_id) if pred_track_id is not None else None
    frame_record = track_info.frame_role_record.get((frame, pred_fighter), {})
    frame_red = float(frame_record.get("score_red", 0.0))
    frame_blue = float(frame_record.get("score_blue", 0.0))
    frame_color_label = "red" if frame_red >= frame_blue else "blue"
    gt_role_track_id = None
    frame_error = ""
    abs_frame_error = ""
    time_error_s = ""
    scorable = ""
    match_type = "fp"
    if gt is not None:
        gt_frame = as_int(gt["frame"], "frame")
        frame_error_int = frame - gt_frame
        time_error = abs(frame_error_int) / FPS
        is_scorable = time_error / 0.5 < 1.0
        gt_role_track_id = track_info.frame_role_track.get((frame, gt["fighter"]))
        frame_error = frame_error_int
        abs_frame_error = abs(frame_error_int)
        time_error_s = f"{time_error:.6f}"
        scorable = str(is_scorable).lower()
        match_type = "tp_scorable" if is_scorable else "tp_time_only"

    role_rule = suggest_track_role(pred, track_info)
    color_rule = suggest_track_color(pred, track_info)
    frame_rule = suggest_frame_color(pred, track_info)
    role_color_rule = role_rule if role_rule == color_rule else None
    row = {
        "video_key": pred["video_key"],
        "data_root": video["data_root"],
        "fight_index": video["fight_index"],
        "round_number": video["round_number"],
        "row_id": pred["id"],
        "pred_index": pred_index,
        "match_type": match_type,
        "pred_frame": pred["frame"],
        "gt_frame": gt["frame"] if gt else "",
        "frame_error": frame_error,
        "abs_frame_error": abs_frame_error,
        "time_error_s": time_error_s,
        "scorable": scorable,
        "pred_fighter": pred["fighter"],
        "gt_fighter": gt["fighter"] if gt else "",
        "fighter_correct": str(pred["fighter"] == gt["fighter"]).lower() if gt else "",
        "pred_hand": pred["hand"],
        "gt_hand": gt["hand"] if gt else "",
        "hand_correct": str(pred["hand"] == gt["hand"]).lower() if gt else "",
        "pred_target": pred["target"],
        "gt_target": gt["target"] if gt else "",
        "target_correct": str(pred["target"] == gt["target"]).lower() if gt else "",
        "pred_punch_type": pred["punch_type"],
        "gt_punch_type": gt["punch_type"] if gt else "",
        "pred_effectiveness": pred["effectiveness"],
        "gt_effectiveness": gt["effectiveness"] if gt else "",
        "pred_track_id": pred_track_id if pred_track_id is not None else "",
        "gt_role_track_id": gt_role_track_id if gt_role_track_id is not None else "",
        "pred_gt_tracks_equal": str(pred_track_id == gt_role_track_id).lower()
        if pred_track_id is not None and gt_role_track_id is not None
        else "",
        "pred_role_majority": summary.role_majority if summary else "",
        "pred_role_purity": f"{summary.role_purity:.6f}" if summary else "",
        "pred_color_label": summary.color_label if summary else "",
        "pred_color_margin": f"{summary.color_margin:.6f}" if summary else "",
        "pred_red_mean": f"{summary.red_mean:.6f}" if summary else "",
        "pred_blue_mean": f"{summary.blue_mean:.6f}" if summary else "",
        "pred_frame_color_label": frame_color_label if frame_record else "",
        "pred_frame_color_margin": f"{frame_red - frame_blue:.6f}" if frame_record else "",
        "pred_track_frames": summary.n_frames if summary else "",
        "pred_track_first": summary.first_frame if summary else "",
        "pred_track_last": summary.last_frame if summary else "",
        "rule_track_role": role_rule or "",
        "rule_track_color": color_rule or "",
        "rule_frame_color": frame_rule or "",
        "rule_role_color_agree": role_color_rule or "",
    }
    return row


def print_score(label: str, score: dict[str, Any], n_rows: int, wins: int) -> None:
    by_video = score["by_video"]
    fighter = np.mean([item["score_fighter"] for item in by_video.values()])
    timing = np.mean([item["score_time"] for item in by_video.values()])
    fp = np.mean([item["fp_penalty"] for item in by_video.values()])
    print(
        f"{label}: score={score['macro_score']:.6f},fighter={fighter:.6f},"
        f"time={timing:.6f},fp={fp:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def print_identity_summary(
    audit_rows: list[dict[str, object]],
    baseline: dict[str, Any],
    video_by_key: dict[str, dict[str, str]],
    top_k: int,
) -> None:
    scorable = [row for row in audit_rows if row["match_type"] == "tp_scorable"]
    wrong = [row for row in scorable if row["fighter_correct"] == "false"]
    print(
        f"matched_scorable={len(scorable)} fighter_wrong={len(wrong)} "
        f"wrong_rate={len(wrong) / max(1, len(scorable)):.4f}"
    )

    print("video_key,root,n_gt,n_pred,n_tp,n_fp,score,fighter,time,wrong_fighter_scorable")
    wrong_by_video = Counter(row["video_key"] for row in wrong)
    for key, item in sorted(
        baseline["by_video"].items(),
        key=lambda kv: (-wrong_by_video[kv[0]], kv[0]),
    )[:top_k]:
        print(
            f"{key},{video_by_key[key]['data_root']},{item['n_gt']},{item['n_pred']},"
            f"{item['n_tp']},{item['n_fp']},{item['final_score']:.6f},"
            f"{item['score_fighter']:.6f},{item['score_time']:.6f},"
            f"{wrong_by_video[key]}"
        )

    print("group,total,wrong,wrong_rate")
    for label, key_fn in [
        ("pred_fighter", lambda row: row["pred_fighter"]),
        ("data_root", lambda row: row["data_root"]),
        ("pred_role_majority", lambda row: row["pred_role_majority"]),
        ("pred_color_label", lambda row: row["pred_color_label"]),
        ("role_color_pair", lambda row: f"{row['pred_role_majority']}->{row['pred_color_label']}"),
    ]:
        totals = Counter(str(key_fn(row)) for row in scorable)
        wrongs = Counter(str(key_fn(row)) for row in wrong)
        for value, total in totals.most_common(top_k):
            print(f"{label}:{value},{total},{wrongs[value]},{wrongs[value] / max(1, total):.4f}")


def print_rule_scores(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    track_info_by_key: dict[str, TrackInfo],
    baseline: dict[str, Any],
    top_k: int,
) -> None:
    rules = build_rules()
    results = []
    for rule in rules:
        rows, n_changed = apply_rule(pred_rows, track_info_by_key, rule)
        score = score_predictions(gt_rows, rows)
        by_video = score["by_video"]
        fighter = np.mean([item["score_fighter"] for item in by_video.values()])
        timing = np.mean([item["score_time"] for item in by_video.values()])
        fp = np.mean([item["fp_penalty"] for item in by_video.values()])
        results.append(
            {
                "rule": rule.name,
                "score": score["macro_score"],
                "delta": score["macro_score"] - baseline["macro_score"],
                "fighter": fighter,
                "time": timing,
                "fp": fp,
                "wins": video_wins(score, baseline),
                "n_changed": n_changed,
            }
        )
    print("rule,score,delta,fighter,time,fp,wins,n_changed")
    for item in sorted(results, key=lambda row: (row["score"], -row["n_changed"]), reverse=True)[:top_k]:
        print(
            f"{item['rule']},{item['score']:.6f},{item['delta']:.6f},"
            f"{item['fighter']:.6f},{item['time']:.6f},{item['fp']:.6f},"
            f"{item['wins']},{item['n_changed']}"
        )


def build_rules() -> list[Rule]:
    rules = [
        Rule("track_role_majority", suggest_track_role),
        Rule("track_color_mean", suggest_track_color),
        Rule("frame_color", suggest_frame_color),
        Rule("track_role_color_agree", suggest_track_role_color_agree),
    ]
    for margin in [0.05, 0.10, 0.20, 0.35]:
        rules.append(
            Rule(
                f"track_color_margin_ge_{margin:.2f}",
                lambda row, info, margin=margin: suggest_track_color(row, info, margin),
            )
        )
        rules.append(
            Rule(
                f"frame_color_margin_ge_{margin:.2f}",
                lambda row, info, margin=margin: suggest_frame_color(row, info, margin),
            )
        )
    for purity in [0.55, 0.65, 0.75]:
        for margin in [0.05, 0.10, 0.20]:
            rules.append(
                Rule(
                    f"low_role_purity_{purity:.2f}_color_{margin:.2f}",
                    lambda row, info, purity=purity, margin=margin: suggest_low_purity_color(
                        row, info, purity, margin
                    ),
                )
            )
    return rules


def apply_rule(
    rows: list[dict[str, str]],
    track_info_by_key: dict[str, TrackInfo],
    rule: Rule,
) -> tuple[list[dict[str, str]], int]:
    output = []
    n_changed = 0
    for row in rows:
        new_row = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        suggestion = rule.suggest(row, track_info_by_key[row["video_key"]])
        if suggestion in FIGHTERS and suggestion != row["fighter"]:
            new_row["fighter"] = suggestion
            n_changed += 1
        output.append(new_row)
    return output, n_changed


def suggest_track_role(row: dict[str, str], info: TrackInfo, margin: float | None = None) -> str | None:
    del margin
    summary = row_track_summary(row, info)
    return summary.role_majority if summary else None


def suggest_track_color(
    row: dict[str, str],
    info: TrackInfo,
    margin: float | None = None,
) -> str | None:
    summary = row_track_summary(row, info)
    if summary is None:
        return None
    if margin is not None and abs(summary.color_margin) < margin:
        return None
    return summary.color_label


def suggest_frame_color(
    row: dict[str, str],
    info: TrackInfo,
    margin: float | None = None,
) -> str | None:
    record = row_frame_record(row, info)
    if not record:
        return None
    red = float(record.get("score_red", 0.0))
    blue = float(record.get("score_blue", 0.0))
    if margin is not None and abs(red - blue) < margin:
        return None
    return "red" if red >= blue else "blue"


def suggest_track_role_color_agree(row: dict[str, str], info: TrackInfo) -> str | None:
    role = suggest_track_role(row, info)
    color = suggest_track_color(row, info)
    return role if role == color else None


def suggest_low_purity_color(
    row: dict[str, str],
    info: TrackInfo,
    max_purity: float,
    min_margin: float,
) -> str | None:
    summary = row_track_summary(row, info)
    if summary is None:
        return None
    if summary.role_purity > max_purity or abs(summary.color_margin) < min_margin:
        return None
    return summary.color_label


def row_track_summary(row: dict[str, str], info: TrackInfo) -> TrackSummary | None:
    frame = as_int(row["frame"], "frame")
    track_id = info.frame_role_track.get((frame, row["fighter"]))
    return info.summaries.get(track_id) if track_id is not None else None


def row_frame_record(row: dict[str, str], info: TrackInfo) -> dict[str, Any] | None:
    frame = as_int(row["frame"], "frame")
    return info.frame_role_record.get((frame, row["fighter"]))


def default_output_path(predictions: Path) -> Path:
    return Path("data/processed/diagnostics") / f"fighter_identity_errors_{predictions.stem}.csv"


if __name__ == "__main__":
    raise SystemExit(main())
