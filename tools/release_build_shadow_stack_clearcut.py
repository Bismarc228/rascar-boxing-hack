#!/usr/bin/env python3
"""Rebuild and verify release candidate RC1: Shadow Stack ClearCut."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

OOF_STACK = Path(
    "data/processed/vit_features/"
    "component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv"
)
STRICT_CURRENT_PKEEP = Path(
    "data/processed/diagnostics/"
    "candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv"
)
OOF_THRESHOLDS = Path(
    "data/processed/diagnostics/"
    "clear_ranked_on_private_stack_transitionopt_thresholds_20260523.csv"
)
OOF_VIDEO = Path(
    "data/processed/diagnostics/"
    "clear_ranked_on_private_stack_transitionopt_video_20260523.csv"
)
OOF_RANKS = Path(
    "data/processed/diagnostics/"
    "clear_ranked_on_private_stack_transitionopt_rows_20260523.csv"
)
PARENT_SUBMISSION = Path(
    "submissions/"
    "prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_"
    "rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv"
)
OUTPUT_DIAGNOSTICS = Path(
    "data/processed/diagnostics/"
    "candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_"
    "e30_top8_blockedmiss_20260523.csv"
)
OUTPUT_SUBMISSION = Path(
    "submissions/"
    "prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_"
    "candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_"
    "OFFLINE_CANDIDATE.csv"
)
TRAIN_PREDICTIONS = Path(
    "data/processed/vit_features/"
    "clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_"
    "posehgb094_vitlogreg012_rows_20260522.csv"
)
TRAIN_TRACKS_DIR = Path("data/processed/pose_tracks/val_yolo26l_conf035")
TEST_TRACKS_DIR = Path("data/processed/pose_tracks/test_yolo26l_conf035")

EXPECTED_SUBMISSION_SHA256 = (
    "da285e84174f8cf9489d09f7d57c8e67c1c052ec385053af47786b79de07262a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip exact SHA-256 assertion and keep metric/validation checks only.",
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
            OOF_STACK,
            STRICT_CURRENT_PKEEP,
            PARENT_SUBMISSION,
            TRAIN_PREDICTIONS,
            TRAIN_TRACKS_DIR,
            TEST_TRACKS_DIR,
            Path("tools/evaluate_clear_precision_ranked_gate.py"),
            Path("tools/evaluate_candidate_token_reranker.py"),
            Path("tools/make_candidate_token_clear_submission.py"),
        ]
    )

    run(
        [
            sys.executable,
            "tools/evaluate_clear_precision_ranked_gate.py",
            "--data-root",
            str(args.data_root),
            "--predictions",
            str(OOF_STACK),
            "--no-default-pkeep-source",
            "--pkeep-source",
            f"strict_current={STRICT_CURRENT_PKEEP}",
            "--thresholds",
            "1,2,3,4,5,6,7,8,9,10,12,15",
            "--top-k",
            "12",
            "--output-thresholds",
            str(OOF_THRESHOLDS),
            "--output-video",
            str(OOF_VIDEO),
            "--output-ranks",
            str(OOF_RANKS),
        ]
    )
    assert_oof_audit(OOF_THRESHOLDS)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    run(
        [
            sys.executable,
            "tools/make_candidate_token_clear_submission.py",
            "--data-root",
            str(args.data_root),
            "--train-predictions",
            str(TRAIN_PREDICTIONS),
            "--test-submission",
            str(PARENT_SUBMISSION),
            "--train-tracks-dir",
            str(TRAIN_TRACKS_DIR),
            "--test-tracks-dir",
            str(TEST_TRACKS_DIR),
            "--epochs",
            str(args.epochs),
            "--feature-mode",
            "pose_audio",
            "--drop-top-k",
            "8",
            "--drop-effectiveness",
            "blocked,miss",
            "--device",
            args.device,
            "--output-diagnostics",
            str(OUTPUT_DIAGNOSTICS),
            "--output-submission",
            str(OUTPUT_SUBMISSION),
        ],
        env=env,
    )

    run(
        [
            sys.executable,
            "tools/validate_data.py",
            "--data-root",
            str(args.data_root),
            "--submission",
            str(OUTPUT_SUBMISSION),
        ]
    )
    assert_submission_diff(PARENT_SUBMISSION, OUTPUT_SUBMISSION)
    if not args.skip_hash_check:
        assert_sha256(OUTPUT_SUBMISSION, EXPECTED_SUBMISSION_SHA256)

    print("RC1 Shadow Stack ClearCut reproduced")
    print(f"submission={OUTPUT_SUBMISSION}")
    print(f"oof_thresholds={OOF_THRESHOLDS}")
    return 0


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def require_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"missing required RC1 input files:\n{joined}")


def assert_oof_audit(path: Path) -> None:
    rows = read_rows(path)
    target = None
    for row in rows:
        if (
            row["model"] == "pkeep_strict_current"
            and row["guard"] == "blocked_miss"
            and row["threshold_type"] == "top_k"
            and row["threshold"] == "8"
        ):
            target = row
            break
    if target is None:
        raise SystemExit("missing OOF audit row: pkeep_strict_current blocked_miss top_k=8")
    expected = {
        "macro_score": "0.420768",
        "n_kept": "1180",
        "n_dropped": "8",
        "dropped_fp": "2",
        "dropped_tp_scorable": "4",
        "dropped_tp_time_only": "2",
        "fp_penalty": "0.059184",
    }
    for key, value in expected.items():
        if target[key] != value:
            raise SystemExit(f"OOF audit mismatch for {key}: got {target[key]!r}, expected {value!r}")


def assert_submission_diff(parent_path: Path, candidate_path: Path) -> None:
    parent = {row["id"]: row for row in read_rows(parent_path)}
    candidate = {row["id"]: row for row in read_rows(candidate_path)}
    dropped = [
        parent[row_id]
        for row_id, row in parent.items()
        if row.get("clear") == "true" and candidate[row_id].get("clear") == "false"
    ]
    if len(dropped) != 8:
        raise SystemExit(f"expected 8 clear drops, got {len(dropped)}")
    by_video = Counter(row["video_key"] for row in dropped)
    by_effectiveness = Counter(row["effectiveness"] for row in dropped)
    if dict(sorted(by_video.items())) != {"agn_037": 1, "agn_038": 1, "agn_047": 6}:
        raise SystemExit(f"unexpected dropped_by_video={dict(sorted(by_video.items()))}")
    if dict(sorted(by_effectiveness.items())) != {"blocked": 2, "miss": 6}:
        raise SystemExit(
            f"unexpected dropped_by_effectiveness={dict(sorted(by_effectiveness.items()))}"
        )


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
