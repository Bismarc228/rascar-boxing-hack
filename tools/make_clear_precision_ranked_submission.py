#!/usr/bin/env python3
"""Train-all HGB clear-risk ranker and materialize top-K test drops."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from tools.evaluate_clear_precision_ranked_gate import (
    GUARDS,
    build_labels,
    build_ranker_matrix,
    load_pkeep_sources,
    parse_source,
)


DEFAULT_TRAIN_PREDICTIONS = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_"
    "posehgb094_vitlogreg012_rows_20260522.csv"
)
DEFAULT_TEST_SUBMISSION = Path(
    "submissions/"
    "prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_"
    "clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, default=DEFAULT_TRAIN_PREDICTIONS)
    parser.add_argument("--test-submission", type=Path, default=DEFAULT_TEST_SUBMISSION)
    parser.add_argument(
        "--train-pkeep-source",
        action="append",
        required=True,
        help="Name/path pair like name=path.csv with train OOF p_keep values.",
    )
    parser.add_argument(
        "--test-pkeep-source",
        action="append",
        required=True,
        help="Name/path pair like name=path.csv with test p_keep values aligned by id.",
    )
    parser.add_argument("--ranker-feature-set", choices=["base", "local"], default="base")
    parser.add_argument("--guard", choices=sorted(GUARDS), default="miss")
    parser.add_argument("--drop-top-k", type=int, required=True)
    parser.add_argument("--protect-video-keys", default="")
    parser.add_argument("--output-diagnostics", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = [row for row in read_csv_rows(args.train_predictions) if row.get("clear") == "true"]
    test_rows_all = read_csv_rows(args.test_submission)
    test_clear_rows = [row for row in test_rows_all if row.get("clear") == "true"]
    if not train_rows:
        raise SystemExit(f"no clear train rows in {args.train_predictions}")
    if not test_clear_rows:
        raise SystemExit(f"no clear test rows in {args.test_submission}")

    train_videos = read_csv_rows(args.data_root / "train/videos.csv")
    test_videos = read_csv_rows(args.data_root / "test/videos.csv")
    train_video_by_key = {row["video_key"]: row for row in train_videos}
    test_video_by_key = {row["video_key"]: row for row in test_videos}
    train_video_keys = {row["video_key"] for row in train_rows}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in train_video_keys
    ]
    _labels_y, label_names = build_labels(train_rows, gt_rows)
    y = np.asarray([label == "fp" for label in label_names], dtype=np.int8)

    train_sources = [parse_source(item) for item in args.train_pkeep_source]
    train_pkeep, match_stats = load_pkeep_sources(train_rows, train_sources)
    test_pkeep = load_test_pkeep_sources(test_clear_rows, [parse_source(item) for item in args.test_pkeep_source])
    if list(train_pkeep) != list(test_pkeep):
        raise SystemExit(
            "train/test p_keep source names differ: "
            f"train={','.join(train_pkeep)} test={','.join(test_pkeep)}"
        )

    x_train, numeric_count = build_ranker_matrix(
        train_rows,
        train_video_by_key,
        train_pkeep,
        args.ranker_feature_set,
    )
    x_test, _test_numeric_count = build_ranker_matrix(
        test_clear_rows,
        test_video_by_key,
        test_pkeep,
        args.ranker_feature_set,
    )
    model = build_model(numeric_count)
    model.fit(x_train, y)
    risk = model.predict_proba(x_test)[:, 1].astype(np.float32)

    drop_ids = select_drop_ids(test_clear_rows, risk, args)
    diagnostics = build_diagnostics(test_clear_rows, test_pkeep, risk, drop_ids)
    write_csv_rows(args.output_diagnostics, diagnostics, diagnostics_columns(test_pkeep))

    output_rows = []
    for row in test_rows_all:
        out = {column: row.get(column, "") for column in SUBMISSION_COLUMNS}
        if out.get("id") in drop_ids and out.get("clear") == "true":
            out["clear"] = "false"
        output_rows.append(out)
    write_csv_rows(args.output_submission, output_rows, SUBMISSION_COLUMNS)

    dropped_rows = [row for row in test_clear_rows if row["id"] in drop_ids]
    print(f"train_clear_rows={len(train_rows)} test_clear_rows={len(test_clear_rows)}")
    print("labels=" + ",".join(f"{key}:{value}" for key, value in Counter(label_names).most_common()))
    for name, stats in match_stats.items():
        print(
            f"train_source={name} exact={stats['exact']} relaxed={stats['relaxed']} "
            f"missing={stats['missing']} duplicate_relaxed={stats['duplicate_relaxed']}"
        )
    print(
        f"ranker=hgb_pkeep feature_set={args.ranker_feature_set} guard={args.guard} "
        f"top_k={args.drop_top_k} dropped={len(drop_ids)}"
    )
    print("dropped_by_video=" + format_counts(Counter(row["video_key"] for row in dropped_rows)))
    print("dropped_by_effectiveness=" + format_counts(Counter(row["effectiveness"] for row in dropped_rows)))
    print(f"diagnostics={args.output_diagnostics}")
    print(f"submission={args.output_submission}")
    return 0


def build_model(numeric_count: int):
    numeric_indices = list(range(numeric_count))
    categorical_indices = None if numeric_count == 0 else list(range(numeric_count, numeric_count + 7))
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_indices,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ]
    )
    return make_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=7,
        ),
    )


def load_test_pkeep_sources(
    rows: list[dict[str, str]],
    sources: list[tuple[str, Path]],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    row_ids = [row["id"] for row in rows]
    for name, path in sources:
        source_rows = read_csv_rows(path)
        if not source_rows or "p_keep" not in source_rows[0]:
            raise SystemExit(f"missing p_keep rows in {path}")
        by_id = {row["id"]: float(row["p_keep"]) for row in source_rows if row.get("id")}
        missing = [row_id for row_id in row_ids if row_id not in by_id]
        if missing:
            raise SystemExit(f"test p_keep source {path} missing {len(missing)} ids")
        output[name] = np.asarray([by_id[row_id] for row_id in row_ids], dtype=np.float32)
    return output


def select_drop_ids(rows: list[dict[str, str]], risk: np.ndarray, args: argparse.Namespace) -> set[str]:
    allowed_effectiveness = GUARDS[args.guard]
    protected = {item.strip() for item in args.protect_video_keys.split(",") if item.strip()}
    eligible = [
        (index, row, float(risk[index]))
        for index, row in enumerate(rows)
        if row.get("video_key") not in protected
        and (allowed_effectiveness is None or row.get("effectiveness", "") in allowed_effectiveness)
    ]
    selected = sorted(eligible, key=lambda item: (-item[2], item[0]))[: args.drop_top_k]
    return {row["id"] for _index, row, _score in selected}


def build_diagnostics(
    rows: list[dict[str, str]],
    pkeep: dict[str, np.ndarray],
    risk: np.ndarray,
    drop_ids: set[str],
) -> list[dict[str, object]]:
    output = []
    for index, row in enumerate(rows):
        item: dict[str, object] = {
            "id": row["id"],
            "video_key": row["video_key"],
            "frame": row["frame"],
            "fighter": row["fighter"],
            "punch_type": row["punch_type"],
            "hand": row["hand"],
            "target": row["target"],
            "effectiveness": row["effectiveness"],
            "risk_fp": f"{float(risk[index]):.9f}",
            "drop": int(row["id"] in drop_ids),
        }
        for name, values in pkeep.items():
            item[f"p_keep_{name}"] = f"{float(values[index]):.9f}"
        output.append(item)
    return output


def diagnostics_columns(pkeep: dict[str, np.ndarray]) -> list[str]:
    return [
        "id",
        "video_key",
        "frame",
        "fighter",
        "punch_type",
        "hand",
        "target",
        "effectiveness",
        "risk_fp",
        *[f"p_keep_{name}" for name in pkeep],
        "drop",
    ]


def format_counts(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


if __name__ == "__main__":
    raise SystemExit(main())
