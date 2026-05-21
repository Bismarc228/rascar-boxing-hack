#!/usr/bin/env python3
"""Evaluate learned per-video base-vs-override policy on validation rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rascar_boxing.io import read_csv_rows
from rascar_boxing.metric import score_predictions
from tools.gate_submission_hybrid import clear_rows_by_video, evaluate_video, split_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/raw/train/punches.csv"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--video-keys", help="Optional comma-separated video keys")
    parser.add_argument("--video-keys-from-predictions", action="store_true")
    parser.add_argument("--cv", choices=["video", "fight"], default="fight")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decision-threshold", type=float, default=0.0)
    parser.add_argument("--max-count-delta", type=int, default=48)
    parser.add_argument("--min-base-within-15", type=float, default=0.85)
    parser.add_argument("--min-override-within-15", type=float, default=0.70)
    parser.add_argument("--max-median-gap", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    gt_rows = read_csv_rows(args.ground_truth)
    base_rows = read_csv_rows(args.base)
    override_rows = read_csv_rows(args.override)
    keys = split_keys(args.video_keys)
    if args.video_keys_from_predictions:
        keys |= {row["video_key"] for row in base_rows}
        keys |= {row["video_key"] for row in override_rows}
    if keys:
        gt_rows = [row for row in gt_rows if row["video_key"] in keys]
        base_rows = [row for row in base_rows if row["video_key"] in keys]
        override_rows = [row for row in override_rows if row["video_key"] in keys]

    base_score = score_predictions(gt_rows, base_rows)
    override_score = score_predictions(gt_rows, override_rows)
    base_by_video = clear_rows_by_video(base_rows)
    override_by_video = clear_rows_by_video(override_rows)
    gate_args = SimpleNamespace(
        max_count_delta=args.max_count_delta,
        min_base_within_15=args.min_base_within_15,
        min_override_within_15=args.min_override_within_15,
        max_median_gap=args.max_median_gap,
    )
    items = []
    for key in sorted(set(base_by_video) | set(override_by_video)):
        if key not in base_score["by_video"] or key not in override_score["by_video"]:
            continue
        video = video_by_key[key]
        gate = evaluate_video(key, base_by_video.get(key, []), override_by_video.get(key, []), gate_args)
        base_value = base_score["by_video"][key]["final_score"]
        override_value = override_score["by_video"][key]["final_score"]
        items.append(
            {
                "key": key,
                "group": fight_group(video) if args.cv == "fight" else key,
                "features": feature_dict(video, gate),
                "delta": override_value - base_value,
                "gate_pick": gate.pass_gate,
                "oracle_pick": override_value > base_value,
            }
        )

    predicted_pick: dict[str, bool] = {}
    for group in sorted({item["group"] for item in items}):
        train = [item for item in items if item["group"] != group]
        valid = [item for item in items if item["group"] == group]
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), Ridge(alpha=args.alpha))
        model.fit([item["features"] for item in train], [item["delta"] for item in train])
        preds = model.predict([item["features"] for item in valid])
        for item, pred in zip(valid, preds):
            item["pred_delta"] = float(pred)
            predicted_pick[item["key"]] = float(pred) > args.decision_threshold

    gate_pick = {item["key"]: item["gate_pick"] for item in items}
    oracle_pick = {item["key"]: item["oracle_pick"] for item in items}
    policies = {
        "base": {},
        "override": {item["key"]: True for item in items},
        "threshold_gate": gate_pick,
        "ridge_policy": predicted_pick,
        "oracle": oracle_pick,
    }
    print(f"base_score={base_score['macro_score']:.6f} override_score={override_score['macro_score']:.6f}")
    print("policy,score,n_rows,replace_keys")
    for name, picks in policies.items():
        rows = hybrid_rows(base_rows, override_rows, {key for key, pick in picks.items() if pick})
        score = score_predictions(gt_rows, rows)
        print(f"{name},{score['macro_score']:.6f},{len(rows)},{','.join(sorted(key for key, pick in picks.items() if pick))}")

    print("video_key,group,delta,pred_delta,gate_pick,ridge_pick,oracle_pick")
    for item in items:
        print(
            f"{item['key']},{item['group']},{item['delta']:+.6f},"
            f"{item.get('pred_delta', 0.0):+.6f},{int(item['gate_pick'])},"
            f"{int(predicted_pick[item['key']])},{int(item['oracle_pick'])}"
        )
    return 0


def feature_dict(video: dict[str, str], gate) -> dict[str, object]:
    round_number = int(video["round_number"] or 0)
    fight_index = int(video["fight_index"] or 0)
    return {
        "data_root=" + video["data_root"]: 1.0,
        "dataset_type=" + video["dataset_type"]: 1.0,
        "fight_index": float(fight_index),
        "round_number": float(round_number),
        "base_count": float(gate.base_count),
        "override_count": float(gate.override_count),
        "count_delta": float(gate.count_delta),
        "abs_count_delta": float(abs(gate.count_delta)),
        "base_within_15": gate.base_within_15,
        "base_within_30": gate.base_within_30,
        "base_within_90": gate.base_within_90,
        "override_within_15": gate.override_within_15,
        "override_within_30": gate.override_within_30,
        "override_within_90": gate.override_within_90,
        "base_median_gap": gate.base_median_gap,
        "override_median_gap": gate.override_median_gap,
    }


def fight_group(video: dict[str, str]) -> str:
    return "|".join([video["data_root"], video["fight_index"], video["fight_folder"]])


def hybrid_rows(
    base_rows: list[dict[str, str]],
    override_rows: list[dict[str, str]],
    replace_keys: set[str],
) -> list[dict[str, str]]:
    rows = [dict(row) for row in base_rows if row["video_key"] not in replace_keys]
    rows.extend(dict(row) for row in override_rows if row["video_key"] in replace_keys)
    rows.sort(key=lambda row: (row["video_key"], int(row["frame"]), row["fighter"], row["hand"]))
    for index, row in enumerate(rows, start=1):
        row["id"] = str(index)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
