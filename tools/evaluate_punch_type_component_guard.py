#!/usr/bin/env python3
"""Guarded punch_type-only component comparison for fixed validation rows."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rascar_boxing.constants import FPS, SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions


PUNCH_CLASSES = ["jab", "cross", "hook", "uppercut"]
MISSED = "missed"

STRICT_ANCHOR = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_"
    "posehgb094_vitlogreg012_rows_20260522.csv"
)
CLIP_M02_ROWS = Path(
    "data/processed/vit_features/"
    "clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_"
    "ptype_m02_rows_20260522.csv"
)
CLIP_M028_ROWS = Path(
    "data/processed/vit_features/"
    "clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_"
    "ptype_m028_rows_20260522.csv"
)
CANDIDATE_TOKEN_ROWS = Path(
    "data/processed/validation_rows/candidate_token_punch_type_margin095_20260522_oof.csv"
)
CANDIDATE_TOKEN_DIAGNOSTIC = Path(
    "data/processed/diagnostics/candidate_token_punch_type_pose_audio_rgb_e40_20260522.csv"
)
ROOTOUT_DIR = Path(
    "data/processed/root_out/"
    "yolo11s_pose_geom_id_attr_model_pose_logreg_noctx_c100_joint_margin_20260521"
)
ROOTOUT_VARIANT = "punch_type_margin_0.1"


@dataclass
class EvalResult:
    split: str
    variant: str
    source: str
    rows: list[dict[str, str]]
    score: dict[str, object]
    summary: dict[str, float]
    confusion: dict[str, Counter[str]]
    recalls: dict[str, float]
    root_scores: dict[str, dict[str, float]]
    n_changed: int
    n_missing_source: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=Path("data/raw/train/punches.csv"))
    parser.add_argument("--videos", type=Path, default=Path("data/raw/train/videos.csv"))
    parser.add_argument("--strict-anchor", type=Path, default=STRICT_ANCHOR)
    parser.add_argument("--clip-m02-rows", type=Path, default=CLIP_M02_ROWS)
    parser.add_argument("--clip-m028-rows", type=Path, default=CLIP_M028_ROWS)
    parser.add_argument("--candidate-token-rows", type=Path, default=CANDIDATE_TOKEN_ROWS)
    parser.add_argument("--candidate-token-diagnostic", type=Path, default=CANDIDATE_TOKEN_DIAGNOSTIC)
    parser.add_argument("--rootout-dir", type=Path, default=ROOTOUT_DIR)
    parser.add_argument("--rootout-variant", default=ROOTOUT_VARIANT)
    parser.add_argument(
        "--token-margins",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99",
    )
    parser.add_argument(
        "--feature-output-dir",
        type=Path,
        default=Path("data/processed/vit_features"),
    )
    parser.add_argument(
        "--diagnostic-output-dir",
        type=Path,
        default=Path("data/processed/diagnostics"),
    )
    parser.add_argument("--stamp", default="20260523")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_rows = [row for row in read_csv_rows(args.ground_truth) if row.get("clear") == "true"]
    videos = {row["video_key"]: row for row in read_csv_rows(args.videos)}

    rootout_learned = load_rootout_rows(args.rootout_dir, args.rootout_variant)
    rootout_source = reconstruct_rootout_source(
        rootout_learned,
        args.rootout_dir / "replacement_audit.csv",
        args.rootout_variant,
    )
    verify_only_punch_type_changed(rootout_source, rootout_learned, "rootout_pose_logreg")

    strict_base_rows = normalize_submission_rows(read_csv_rows(args.strict_anchor))
    strict_base = evaluate(
        "strict",
        "clip_margin_m028_anchor",
        "strict anchor already carries CLIP margin 0.28 punch_type",
        strict_base_rows,
        gt_rows,
        videos,
        base_rows=strict_base_rows,
    )

    strict_results = [strict_base]
    strict_results.append(
        evaluate_source_variant(
            "strict",
            "clip_margin_m02_on_anchor",
            str(args.clip_m02_rows),
            strict_base_rows,
            read_csv_rows(args.clip_m02_rows),
            gt_rows,
            videos,
            strict_base_rows,
            strict_base,
            ("id",),
        )
    )
    strict_results.append(
        evaluate_source_variant(
            "strict",
            "clip_margin_m028_on_anchor",
            str(args.clip_m028_rows),
            strict_base_rows,
            read_csv_rows(args.clip_m028_rows),
            gt_rows,
            videos,
            strict_base_rows,
            strict_base,
            ("id",),
        )
    )
    strict_results.append(
        evaluate_source_variant(
            "strict",
            "candidate_token_margin095_source_on_anchor",
            str(args.candidate_token_rows),
            strict_base_rows,
            read_csv_rows(args.candidate_token_rows),
            gt_rows,
            videos,
            strict_base_rows,
            strict_base,
            ("id",),
        )
    )
    strict_results.extend(
        evaluate_token_margin_grid(
            strict_base_rows,
            read_csv_rows(args.candidate_token_diagnostic),
            parse_float_list(args.token_margins),
            gt_rows,
            videos,
            strict_base,
        )
    )
    strict_results.append(
        evaluate_source_variant(
            "strict",
            "pose_logreg_m01_frame_fighter_hand_on_anchor",
            f"{args.rootout_dir}:{args.rootout_variant}",
            strict_base_rows,
            rootout_learned,
            gt_rows,
            videos,
            strict_base_rows,
            strict_base,
            ("video_key", "frame", "fighter", "hand"),
        )
    )

    rootout_base = evaluate(
        "rootout",
        "source_before_pose_logreg",
        f"{args.rootout_dir}/replacement_audit.csv source columns",
        rootout_source,
        gt_rows,
        videos,
        base_rows=rootout_source,
    )
    rootout_pose = evaluate(
        "rootout",
        "pose_logreg_punch_type_margin_0p1",
        f"{args.rootout_dir} top-level {args.rootout_variant} learned rows",
        rootout_learned,
        gt_rows,
        videos,
        base_rows=rootout_source,
        baseline=rootout_base,
    )
    rootout_results = [rootout_base, rootout_pose]

    all_results = strict_results + rootout_results
    write_outputs(args, all_results, strict_base, rootout_base)
    print_console_summary(all_results, strict_base, rootout_base)
    return 0


def normalize_submission_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [{col: row.get(col, "") for col in SUBMISSION_COLUMNS} for row in rows]


def evaluate_source_variant(
    split: str,
    variant: str,
    source: str,
    base_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    videos: dict[str, dict[str, str]],
    guard_base_rows: list[dict[str, str]],
    baseline: EvalResult,
    key_columns: tuple[str, ...],
) -> EvalResult:
    rows, missing = apply_source_punch_type(base_rows, source_rows, key_columns)
    verify_only_punch_type_changed(guard_base_rows, rows, variant)
    return evaluate(
        split,
        variant,
        source,
        rows,
        gt_rows,
        videos,
        base_rows=guard_base_rows,
        baseline=baseline,
        n_missing_source=missing,
    )


def evaluate_token_margin_grid(
    base_rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    margins: list[float],
    gt_rows: list[dict[str, str]],
    videos: dict[str, dict[str, str]],
    baseline: EvalResult,
) -> list[EvalResult]:
    output = []
    for margin in margins:
        rows, missing = apply_token_margin(base_rows, diagnostic_rows, margin)
        variant = f"candidate_token_margin_{format_margin(margin)}_on_anchor"
        verify_only_punch_type_changed(base_rows, rows, variant)
        output.append(
            evaluate(
                "strict",
                variant,
                "candidate token diagnostic top probability minus current anchor probability",
                rows,
                gt_rows,
                videos,
                base_rows=base_rows,
                baseline=baseline,
                n_missing_source=missing,
            )
        )
    return output


def evaluate(
    split: str,
    variant: str,
    source: str,
    rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    videos: dict[str, dict[str, str]],
    base_rows: list[dict[str, str]],
    baseline: EvalResult | None = None,
    n_missing_source: int = 0,
) -> EvalResult:
    keys = {row["video_key"] for row in rows}
    gt_subset = [row for row in gt_rows if row["video_key"] in keys]
    score = score_predictions(gt_subset, rows)
    confusion = punch_type_confusion(gt_subset, rows)
    recalls = recall_by_class(confusion)
    root_scores = score_by_root(gt_subset, rows, videos)
    n_changed = count_punch_type_changes(base_rows, rows)
    return EvalResult(
        split=split,
        variant=variant,
        source=source,
        rows=rows,
        score=score,
        summary=component_summary(score),
        confusion=confusion,
        recalls=recalls,
        root_scores=root_scores,
        n_changed=n_changed,
        n_missing_source=n_missing_source,
    )


def apply_source_punch_type(
    base_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    key_columns: tuple[str, ...],
) -> tuple[list[dict[str, str]], int]:
    source_map, ambiguous = unique_source_map(source_rows, key_columns)
    output = []
    missing = 0
    for row in base_rows:
        item = dict(row)
        key = tuple(row[col] for col in key_columns)
        source_row = source_map.get(key)
        if source_row is None or key in ambiguous:
            missing += 1
        else:
            item["punch_type"] = source_row["punch_type"]
        output.append(item)
    return output, missing


def apply_token_margin(
    base_rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    margin: float,
) -> tuple[list[dict[str, str]], int]:
    diagnostic_by_id = {row["id"]: row for row in diagnostic_rows}
    output = []
    missing = 0
    for row in base_rows:
        item = dict(row)
        diag = diagnostic_by_id.get(row["id"])
        if diag is None:
            missing += 1
            output.append(item)
            continue
        probs = {
            label: float(diag.get(f"p_punch_type_{label}", "0") or 0.0)
            for label in PUNCH_CLASSES
        }
        pred = max(PUNCH_CLASSES, key=lambda label: probs[label])
        current = row["punch_type"]
        current_prob = probs.get(current, 0.0)
        if pred != current and probs[pred] - current_prob >= margin:
            item["punch_type"] = pred
        output.append(item)
    return output, missing


def unique_source_map(
    rows: list[dict[str, str]],
    key_columns: tuple[str, ...],
) -> tuple[dict[tuple[str, ...], dict[str, str]], set[tuple[str, ...]]]:
    source_map: dict[tuple[str, ...], dict[str, str]] = {}
    ambiguous: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[col] for col in key_columns)
        if key in source_map:
            ambiguous.add(key)
        else:
            source_map[key] = row
    return source_map, ambiguous


def load_rootout_rows(rootout_dir: Path, variant: str) -> list[dict[str, str]]:
    paths = sorted(rootout_dir.glob(f"*_{variant}_learned_attr_val_rows.csv"))
    if not paths:
        raise FileNotFoundError(f"No root-out rows found for variant {variant} in {rootout_dir}")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(normalize_submission_rows(read_csv_rows(path)))
    return rows


def reconstruct_rootout_source(
    learned_rows: list[dict[str, str]],
    audit_path: Path,
    variant: str,
) -> list[dict[str, str]]:
    source_rows = [dict(row) for row in learned_rows]
    row_by_key = {(row["video_key"], row["id"]): row for row in source_rows}
    for audit in read_csv_rows(audit_path):
        if audit.get("variant") != variant:
            continue
        row = row_by_key[(audit["video_key"], audit["id"])]
        for column in ["punch_type", "effectiveness", "hand", "target"]:
            if audit.get(f"changed_{column}") == "1":
                row[column] = audit[f"source_{column}"]
    return source_rows


def verify_only_punch_type_changed(
    base_rows: list[dict[str, str]],
    rows: list[dict[str, str]],
    label: str,
) -> None:
    if len(base_rows) != len(rows):
        raise ValueError(f"{label}: row count changed {len(base_rows)} -> {len(rows)}")
    protected = [col for col in SUBMISSION_COLUMNS if col != "punch_type"]
    for index, (base, row) in enumerate(zip(base_rows, rows)):
        for column in protected:
            if base.get(column, "") != row.get(column, ""):
                raise ValueError(
                    f"{label}: non-punch_type column changed at row {index} "
                    f"id={base.get('id')} column={column}: "
                    f"{base.get(column)!r} -> {row.get(column)!r}"
                )


def count_punch_type_changes(
    base_rows: list[dict[str, str]],
    rows: list[dict[str, str]],
) -> int:
    return sum(base["punch_type"] != row["punch_type"] for base, row in zip(base_rows, rows))


def component_summary(score: dict[str, object]) -> dict[str, float]:
    by_video = score["by_video"]
    values = list(by_video.values())
    return {
        "macro": float(score["macro_score"]),
        "time": float(np.mean([item["score_time"] for item in values])),
        "fighter": float(np.mean([item["score_fighter"] for item in values])),
        "punch_type": float(np.mean([item["score_punch_type"] for item in values])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in values])),
        "hand": float(np.mean([item["score_hand"] for item in values])),
        "target": float(np.mean([item["score_target"] for item in values])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in values])),
    }


def punch_type_confusion(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
) -> dict[str, Counter[str]]:
    confusion = {label: Counter() for label in PUNCH_CLASSES}
    gt_by_video = group_by([row for row in gt_rows if row.get("clear") == "true"], "video_key")
    pred_by_video = group_by([row for row in pred_rows if row.get("clear") == "true"], "video_key")
    for video_key, video_gt in gt_by_video.items():
        video_pred = pred_by_video.get(video_key, [])
        assigned: dict[int, str] = {}
        for match in match_events(video_gt, video_pred):
            gt_frame = as_int(video_gt[match.gt_index]["frame"], "frame")
            pred_frame = as_int(video_pred[match.pred_index]["frame"], "frame")
            if abs(gt_frame - pred_frame) / FPS / 0.5 < 1.0:
                assigned[match.gt_index] = video_pred[match.pred_index]["punch_type"]
        for gt_index, gt in enumerate(video_gt):
            gt_label = gt["punch_type"]
            if gt_label not in confusion:
                continue
            pred_label = assigned.get(gt_index, MISSED)
            if pred_label not in PUNCH_CLASSES:
                pred_label = MISSED
            confusion[gt_label][pred_label] += 1
    return confusion


def recall_by_class(confusion: dict[str, Counter[str]]) -> dict[str, float]:
    recalls = {}
    for label in PUNCH_CLASSES:
        total = sum(confusion[label].values())
        recalls[label] = confusion[label][label] / total if total else 0.0
    return recalls


def score_by_root(
    gt_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    videos: dict[str, dict[str, str]],
) -> dict[str, dict[str, float]]:
    keys_by_root: dict[str, set[str]] = {}
    for row in pred_rows:
        video = videos.get(row["video_key"], {})
        root = video.get("data_root", "unknown")
        keys_by_root.setdefault(root, set()).add(row["video_key"])
    output = {}
    for root, keys in sorted(keys_by_root.items()):
        root_gt = [row for row in gt_rows if row["video_key"] in keys]
        root_pred = [row for row in pred_rows if row["video_key"] in keys]
        output[root] = component_summary(score_predictions(root_gt, root_pred))
    return output


def summary_row(
    result: EvalResult,
    baseline: EvalResult,
) -> dict[str, object]:
    macro_delta = result.summary["macro"] - baseline.summary["macro"]
    ptype_delta = result.summary["punch_type"] - baseline.summary["punch_type"]
    jab_delta = result.recalls["jab"] - baseline.recalls["jab"]
    uppercut_delta = result.recalls["uppercut"] - baseline.recalls["uppercut"]
    root_deltas = root_punch_type_deltas(result, baseline)
    min_root_delta = min(root_deltas.values()) if root_deltas else 0.0
    is_baseline = result.variant == baseline.variant and result.split == baseline.split
    pass_score = result.summary["punch_type"] >= 0.235 or macro_delta >= 0.003
    pass_recall = jab_delta > 0.0 and uppercut_delta > 0.0
    pass_root = min_root_delta >= -0.01
    return {
        "split": result.split,
        "variant": result.variant,
        "source": result.source,
        "n_rows": len(result.rows),
        "n_changed": result.n_changed,
        "n_missing_source": result.n_missing_source,
        "macro": f"{result.summary['macro']:.9f}",
        "macro_delta": f"{macro_delta:.9f}",
        "score_punch_type": f"{result.summary['punch_type']:.9f}",
        "punch_type_delta": f"{ptype_delta:.9f}",
        "weighted_punch_type_delta": f"{0.10 * ptype_delta:.9f}",
        "jab_recall": f"{result.recalls['jab']:.9f}",
        "jab_recall_delta": f"{jab_delta:.9f}",
        "uppercut_recall": f"{result.recalls['uppercut']:.9f}",
        "uppercut_recall_delta": f"{uppercut_delta:.9f}",
        "min_root_punch_type_delta": f"{min_root_delta:.9f}",
        "pass_score": "" if is_baseline else str(pass_score).lower(),
        "pass_recall": "" if is_baseline else str(pass_recall).lower(),
        "pass_per_root": "" if is_baseline else str(pass_root).lower(),
        "pass": "" if is_baseline else str(pass_score and pass_recall and pass_root).lower(),
    }


def root_punch_type_deltas(
    result: EvalResult,
    baseline: EvalResult,
) -> dict[str, float]:
    deltas = {}
    for root, item in result.root_scores.items():
        if root not in baseline.root_scores:
            continue
        deltas[root] = item["punch_type"] - baseline.root_scores[root]["punch_type"]
    return deltas


def confusion_rows(result: EvalResult) -> list[dict[str, object]]:
    output = []
    for gt_label in PUNCH_CLASSES:
        counts = result.confusion[gt_label]
        total = sum(counts.values())
        row = {
            "split": result.split,
            "variant": result.variant,
            "gt_punch_type": gt_label,
            "pred_jab": counts["jab"],
            "pred_cross": counts["cross"],
            "pred_hook": counts["hook"],
            "pred_uppercut": counts["uppercut"],
            "missed_or_unscorable": counts[MISSED],
            "total": total,
            "recall": f"{result.recalls[gt_label]:.9f}",
        }
        output.append(row)
    return output


def per_root_rows(result: EvalResult, baseline: EvalResult) -> list[dict[str, object]]:
    output = []
    deltas = root_punch_type_deltas(result, baseline)
    for root, item in result.root_scores.items():
        output.append(
            {
                "split": result.split,
                "variant": result.variant,
                "root": root,
                "macro": f"{item['macro']:.9f}",
                "score_punch_type": f"{item['punch_type']:.9f}",
                "punch_type_delta": f"{deltas.get(root, 0.0):.9f}",
            }
        )
    return output


def change_rows(result: EvalResult, baseline_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    counter: Counter[tuple[str, str]] = Counter()
    for base, row in zip(baseline_rows, result.rows):
        if base["punch_type"] != row["punch_type"]:
            counter[(base["punch_type"], row["punch_type"])] += 1
    return [
        {
            "split": result.split,
            "variant": result.variant,
            "from_punch_type": source,
            "to_punch_type": target,
            "count": count,
        }
        for (source, target), count in sorted(counter.items())
    ]


def write_outputs(
    args: argparse.Namespace,
    all_results: list[EvalResult],
    strict_base: EvalResult,
    rootout_base: EvalResult,
) -> None:
    diag_dir = args.diagnostic_output_dir
    feature_dir = args.feature_output_dir
    summary_path = diag_dir / f"punch_type_component_summary_{args.stamp}.csv"
    confusion_path = diag_dir / f"punch_type_component_confusion_{args.stamp}.csv"
    per_root_path = diag_dir / f"punch_type_component_per_root_{args.stamp}.csv"
    changes_path = diag_dir / f"punch_type_component_changes_{args.stamp}.csv"

    summary_rows = []
    confusion_output = []
    per_root_output = []
    changes_output = []
    for result in all_results:
        baseline = strict_base if result.split == "strict" else rootout_base
        summary_rows.append(summary_row(result, baseline))
        confusion_output.extend(confusion_rows(result))
        per_root_output.extend(per_root_rows(result, baseline))
        changes_output.extend(change_rows(result, baseline.rows))

    write_csv_rows(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_csv_rows(confusion_path, confusion_output, list(confusion_output[0].keys()))
    write_csv_rows(per_root_path, per_root_output, list(per_root_output[0].keys()))
    write_csv_rows(
        changes_path,
        changes_output,
        ["split", "variant", "from_punch_type", "to_punch_type", "count"],
    )

    best_token = max(
        (
            result
            for result in all_results
            if result.variant.startswith("candidate_token_margin_")
            and result.variant.endswith("_on_anchor")
        ),
        key=lambda result: result.summary["macro"],
    )
    strict_pose = next(
        result
        for result in all_results
        if result.variant == "pose_logreg_m01_frame_fighter_hand_on_anchor"
    )
    rootout_pose = next(
        result
        for result in all_results
        if result.variant == "pose_logreg_punch_type_margin_0p1"
    )
    write_csv_rows(
        feature_dir / f"punch_type_component_strict_token_best_rows_{args.stamp}.csv",
        best_token.rows,
        SUBMISSION_COLUMNS,
    )
    write_csv_rows(
        feature_dir / f"punch_type_component_strict_pose_logreg_aligned_rows_{args.stamp}.csv",
        strict_pose.rows,
        SUBMISSION_COLUMNS,
    )
    write_csv_rows(
        feature_dir / f"punch_type_component_rootout_pose_logreg_m01_rows_{args.stamp}.csv",
        rootout_pose.rows,
        SUBMISSION_COLUMNS,
    )


def print_console_summary(
    all_results: list[EvalResult],
    strict_base: EvalResult,
    rootout_base: EvalResult,
) -> None:
    print(
        "split,variant,macro,macro_delta,score_punch_type,punch_type_delta,"
        "jab_recall_delta,uppercut_recall_delta,min_root_punch_type_delta,"
        "n_changed,n_missing_source,pass"
    )
    for result in all_results:
        baseline = strict_base if result.split == "strict" else rootout_base
        row = summary_row(result, baseline)
        print(
            ",".join(
                [
                    str(row["split"]),
                    str(row["variant"]),
                    str(row["macro"]),
                    str(row["macro_delta"]),
                    str(row["score_punch_type"]),
                    str(row["punch_type_delta"]),
                    str(row["jab_recall_delta"]),
                    str(row["uppercut_recall_delta"]),
                    str(row["min_root_punch_type_delta"]),
                    str(row["n_changed"]),
                    str(row["n_missing_source"]),
                    str(row["pass"]),
                ]
            )
        )


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item]


def format_margin(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
