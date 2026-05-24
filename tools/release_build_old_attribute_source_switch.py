#!/usr/bin/env python3
"""Rebuild and verify release candidate RC2: Old-Attribute Source Switch."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

BASE_SOURCE_NAME = "seq_motion"
OVERRIDE_SOURCE_NAME = "old_attr"
SOURCE_LE15_THRESHOLD = 0.93

VALIDATION_BASE = Path(
    "data/processed/validation_rows/"
    "seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv"
)
VALIDATION_OVERRIDE = Path(
    "data/processed/validation_rows/"
    "hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv"
)
VALIDATION_TABLE = Path(
    "data/processed/diagnostics/source_switch_old_attr_validation_release_20260523.csv"
)
VALIDATION_OUTPUT = Path(
    "data/processed/validation_rows/seq_motion_old_attr_sourcele15_093_oof.csv"
)

TEST_BASE = Path(
    "submissions/"
    "seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_"
    "exchange_attr_motion_gate_OFFLINE_CANDIDATE.csv"
)
TEST_OVERRIDE = Path(
    "submissions/"
    "hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_exchange_attr_all_"
    "OFFLINE_CANDIDATE.csv"
)
TEST_TABLE = Path(
    "data/processed/diagnostics/source_switch_old_attr_test_release_20260523.csv"
)
TEST_OUTPUT = Path("submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv")

EXPECTED_VALIDATION_KEYS = [
    "agn_003",
    "agn_024",
    "agn_025",
    "agn_056",
    "agn_057",
    "agn_058",
    "agn_069",
    "agn_070",
    "agn_071",
]
EXPECTED_TEST_KEYS = ["agn_038", "agn_062", "agn_063"]
EXPECTED_VALIDATION_SHA256 = "1123c112665e351ea0249814a19b6d4a01a29d65273a7482fa0d0981515bc731"
EXPECTED_TEST_SHA256 = "a34c7eb6bbe11f1731a586ec8869a413a2ab1dd04c4648efeedb61847139102e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip exact SHA-256 assertions and keep score/validation checks only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_paths(
        [
            args.data_root / "train/punches.csv",
            args.data_root / "train/videos.csv",
            args.data_root / "test/videos.csv",
            args.data_root / "sample_submission.csv",
            VALIDATION_BASE,
            VALIDATION_OVERRIDE,
            TEST_BASE,
            TEST_OVERRIDE,
            Path("tools/build_fight_level_source_table.py"),
            Path("tools/evaluate_source_decision_stumps.py"),
            Path("tools/materialize_source_switch_rows.py"),
            Path("tools/score_predictions.py"),
            Path("tools/validate_data.py"),
        ]
    )

    build_source_table("validation", VALIDATION_BASE, VALIDATION_OVERRIDE, VALIDATION_TABLE, args.data_root)
    stump_stdout = run_capture(
        [
            sys.executable,
            "tools/evaluate_source_decision_stumps.py",
            "--validation-table",
            str(VALIDATION_TABLE),
            "--base-source",
            BASE_SOURCE_NAME,
            "--sources",
            OVERRIDE_SOURCE_NAME,
            "--top-k",
            "3",
        ]
    )
    assert_contains(stump_stdout, "rule=old_attr:source_le15>=0.9394 score=0.411775")
    assert_contains(stump_stdout, "rule=OOF_SELECTED score=0.399638")

    validation_keys = selected_keys(VALIDATION_TABLE)
    assert_keys("validation", validation_keys, EXPECTED_VALIDATION_KEYS)
    materialize(VALIDATION_BASE, VALIDATION_OVERRIDE, validation_keys, VALIDATION_OUTPUT)
    score_stdout = run_capture(
        [
            sys.executable,
            "tools/score_predictions.py",
            "--ground-truth",
            str(args.data_root / "train/punches.csv"),
            "--predictions",
            str(VALIDATION_OUTPUT),
            "--video-keys-from-predictions",
        ]
    )
    assert_contains(score_stdout, "macro_score=0.410780")
    assert_validation_counts(VALIDATION_OUTPUT, n_rows=1263, n_tp=1144, n_fp=119, data_root=args.data_root)

    build_source_table("test", TEST_BASE, TEST_OVERRIDE, TEST_TABLE, args.data_root)
    test_keys = selected_keys(TEST_TABLE)
    assert_keys("test", test_keys, EXPECTED_TEST_KEYS)
    materialize(TEST_BASE, TEST_OVERRIDE, test_keys, TEST_OUTPUT)
    run(
        [
            sys.executable,
            "tools/validate_data.py",
            "--data-root",
            str(args.data_root),
            "--submission",
            str(TEST_OUTPUT),
        ]
    )
    assert_clear_rows(TEST_OUTPUT, expected=734)

    if not args.skip_hash_check:
        assert_sha256(VALIDATION_OUTPUT, EXPECTED_VALIDATION_SHA256)
        assert_sha256(TEST_OUTPUT, EXPECTED_TEST_SHA256)

    print("RC2 Old-Attribute Source Switch reproduced")
    print(f"validation_rows={VALIDATION_OUTPUT}")
    print(f"submission={TEST_OUTPUT}")
    return 0


def build_source_table(split: str, base: Path, override: Path, output: Path, data_root: Path) -> None:
    run(
        [
            sys.executable,
            "tools/build_fight_level_source_table.py",
            "--data-root",
            str(data_root),
            "--split",
            split,
            "--source",
            f"{BASE_SOURCE_NAME}={base}",
            "--source",
            f"{OVERRIDE_SOURCE_NAME}={override}",
            "--output",
            str(output),
        ]
    )


def materialize(base: Path, override: Path, video_keys: list[str], output: Path) -> None:
    run(
        [
            sys.executable,
            "tools/materialize_source_switch_rows.py",
            "--base",
            str(base),
            "--override",
            str(override),
            "--video-keys",
            ",".join(video_keys),
            "--output",
            str(output),
        ]
    )


def selected_keys(table_path: Path) -> list[str]:
    rows = read_rows(table_path)
    keys = [
        row["video_key"]
        for row in rows
        if row["source"] == OVERRIDE_SOURCE_NAME
        and row.get("source_le15")
        and float(row["source_le15"]) >= SOURCE_LE15_THRESHOLD
    ]
    return sorted(keys)


def assert_validation_counts(path: Path, *, n_rows: int, n_tp: int, n_fp: int, data_root: Path) -> None:
    from rascar_boxing.io import read_csv_rows
    from rascar_boxing.metric import score_predictions

    pred_rows = read_csv_rows(ROOT / path)
    gt_rows = [
        row
        for row in read_csv_rows(ROOT / data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in {item["video_key"] for item in pred_rows}
    ]
    score = score_predictions(gt_rows, pred_rows)
    actual_tp = sum(item["n_tp"] for item in score["by_video"].values())
    actual_fp = sum(item["n_fp"] for item in score["by_video"].values())
    if len(pred_rows) != n_rows or actual_tp != n_tp or actual_fp != n_fp:
        raise SystemExit(
            "validation count mismatch: "
            f"rows={len(pred_rows)} tp={actual_tp} fp={actual_fp}, "
            f"expected rows={n_rows} tp={n_tp} fp={n_fp}"
        )


def assert_clear_rows(path: Path, *, expected: int) -> None:
    actual = sum(row.get("clear") == "true" for row in read_rows(path))
    if actual != expected:
        raise SystemExit(f"clear row count mismatch for {path}: got {actual}, expected {expected}")


def assert_keys(label: str, actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise SystemExit(f"{label} selected keys mismatch: got {actual}, expected {expected}")


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing expected output line fragment: {needle}")


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_capture(cmd: list[str]) -> str:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.stdout


def require_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"missing required RC2 input files:\n{joined}")


def assert_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}: got {actual}, expected {expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    raise SystemExit(main())
