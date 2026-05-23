#!/usr/bin/env python3
"""Train-all candidate-token clear inference for fixed-row test submissions.

This is a reproducibility bridge around the historical OOF-only
``evaluate_candidate_token_reranker.py`` from commit d80c86c. It intentionally
keeps the legacy feature/model implementation intact and adds only train-all
test inference plus optional fixed-row clear drops.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows


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
LEGACY_REF = "d80c86c38b12e4b3a0c05126daaa812b8dcaf46e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-ref", default=LEGACY_REF)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-predictions", type=Path, default=DEFAULT_TRAIN_PREDICTIONS)
    parser.add_argument("--test-submission", type=Path, default=DEFAULT_TEST_SUBMISSION)
    parser.add_argument("--train-tracks-dir", type=Path, default=Path("data/processed/pose_tracks/val_yolo26l_conf035"))
    parser.add_argument("--test-tracks-dir", type=Path, default=Path("data/processed/pose_tracks/test_yolo26l_conf035"))
    parser.add_argument("--feature-mode", choices=["pose", "pose_audio"], default="pose_audio")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--pose-feature-windows", default="4,8,16")
    parser.add_argument("--audio-feature-windows", default="0,3,6,12,24")
    parser.add_argument("--audio-offsets", default="-12,-6,-3,0,3,6,12")
    parser.add_argument("--candidate-match-window", type=int, default=6)
    parser.add_argument("--rgb-feature-cache", type=Path)
    parser.add_argument("--soft-time-only", type=float, default=0.40)
    parser.add_argument("--fp-weight", type=float, default=1.15)
    parser.add_argument("--time-only-weight", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.020)
    parser.add_argument(
        "--drop-top-k",
        type=int,
        default=0,
        help="Drop the K lowest-p_keep eligible rows instead of all rows below --threshold.",
    )
    parser.add_argument(
        "--drop-effectiveness",
        default="miss",
        help="Comma-separated effectiveness values eligible for clear=false drops. Empty means all.",
    )
    parser.add_argument("--protect-video-keys", default="")
    parser.add_argument("--output-diagnostics", type=Path, required=True)
    parser.add_argument("--output-submission", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy = load_legacy_module(args.legacy_ref)
    legacy.seed_everything(args.seed)
    device = legacy.pick_device(args.device)

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
    train_keys = sorted({row["video_key"] for row in train_rows})
    test_keys = sorted({row["video_key"] for row in test_clear_rows})
    train_gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(train_keys)
    ]
    _labels_y, label_names = legacy.build_labels(train_rows, train_gt_rows)

    train_pack_args = pack_args(args, args.train_tracks_dir)
    test_pack_args = pack_args(args, args.test_tracks_dir)
    train_sequences = legacy.build_sequences(
        train_pack_args,
        train_rows,
        train_video_by_key,
        train_keys,
        label_names,
        [None] * len(train_rows),
        "clear",
    )
    test_sequences = legacy.build_sequences(
        test_pack_args,
        test_clear_rows,
        test_video_by_key,
        test_keys,
        ["tp_scorable"] * len(test_clear_rows),
        [None] * len(test_clear_rows),
        "clear",
    )

    scaler = legacy.fit_scaler(np.concatenate([seq.x for seq in train_sequences], axis=0))
    train_scaled = [legacy.replace_x(seq, legacy.apply_scaler(seq.x, scaler)) for seq in train_sequences]
    test_scaled = [legacy.replace_x(seq, legacy.apply_scaler(seq.x, scaler)) for seq in test_sequences]
    model = legacy.train_clear_model(args, train_scaled, device)
    p_keep = infer_pkeep(legacy, model, test_scaled, len(test_clear_rows), device)

    drop_ids = select_drop_ids(test_clear_rows, p_keep, args)
    diagnostics = build_diagnostics(test_clear_rows, p_keep, drop_ids)
    write_csv_rows(args.output_diagnostics, diagnostics, diagnostics_columns())

    dropped = 0
    if args.output_submission:
        output_rows = []
        for row in test_rows_all:
            out = {column: row.get(column, "") for column in SUBMISSION_COLUMNS}
            if out.get("clear") == "true" and out.get("id") in drop_ids:
                out["clear"] = "false"
                dropped += 1
            output_rows.append(out)
        write_csv_rows(args.output_submission, output_rows, SUBMISSION_COLUMNS)

    label_counts = {name: label_names.count(name) for name in sorted(set(label_names))}
    print(
        f"legacy_ref={args.legacy_ref} device={device} "
        f"train_clear_rows={len(train_rows)} test_clear_rows={len(test_clear_rows)} "
        f"labels={label_counts} threshold={args.threshold:.6g} top_k={args.drop_top_k} "
        f"drop_effectiveness={args.drop_effectiveness or 'all'} dropped={dropped} "
        f"diagnostics={args.output_diagnostics}"
    )
    if args.output_submission:
        print(f"submission={args.output_submission}")
    return 0


def load_legacy_module(ref: str) -> types.ModuleType:
    local_path = ROOT / "tools/evaluate_candidate_token_reranker.py"
    if local_path.exists():
        spec = importlib.util.spec_from_file_location("_legacy_candidate_token_reranker", local_path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load legacy module from {local_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    code = subprocess.check_output(
        ["git", "show", f"{ref}:tools/evaluate_candidate_token_reranker.py"],
        cwd=ROOT,
        text=True,
    )
    name = "_legacy_candidate_token_reranker"
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / "tools/evaluate_candidate_token_reranker.py")
    sys.modules[name] = module
    exec(compile(code, module.__file__, "exec"), module.__dict__)
    return module


def pack_args(args: argparse.Namespace, tracks_dir: Path) -> argparse.Namespace:
    values = vars(args).copy()
    values["tracks_dir"] = tracks_dir
    return argparse.Namespace(**values)


def infer_pkeep(
    legacy: types.ModuleType,
    model: torch.nn.Module,
    sequences: list[object],
    n_rows: int,
    device: torch.device,
) -> np.ndarray:
    output = np.zeros(n_rows, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            x = torch.from_numpy(seq.x).unsqueeze(0).to(device)
            pad = torch.zeros((1, len(seq.row_indices)), dtype=torch.bool, device=device)
            prob = torch.sigmoid(model(x, pad)).squeeze(0).detach().cpu().numpy().astype(np.float32)
            for row_index, value in zip(seq.row_indices, prob):
                output[row_index] = float(value)
    return output


def build_diagnostics(
    rows: list[dict[str, str]],
    p_keep: np.ndarray,
    drop_ids: set[str],
) -> list[dict[str, object]]:
    output = []
    for row, prob in zip(rows, p_keep):
        output.append(
            {
                "id": row["id"],
                "video_key": row["video_key"],
                "frame": row["frame"],
                "fighter": row["fighter"],
                "punch_type": row["punch_type"],
                "hand": row["hand"],
                "target": row["target"],
                "effectiveness": row["effectiveness"],
                "p_keep": f"{float(prob):.9f}",
                "drop": int(row["id"] in drop_ids),
            }
        )
    return output


def select_drop_ids(
    rows: list[dict[str, str]],
    p_keep: np.ndarray,
    args: argparse.Namespace,
) -> set[str]:
    allowed = split_keys(args.drop_effectiveness)
    protected = split_keys(args.protect_video_keys)
    eligible = [
        (index, row, float(prob))
        for index, (row, prob) in enumerate(zip(rows, p_keep))
        if row.get("video_key") not in protected
        and (not allowed or row.get("effectiveness", "") in allowed)
    ]
    if args.drop_top_k > 0:
        selected = sorted(eligible, key=lambda item: (item[2], item[0]))[: args.drop_top_k]
    else:
        selected = [item for item in eligible if item[2] < args.threshold]
    return {row["id"] for _index, row, _prob in selected}


def split_keys(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def diagnostics_columns() -> list[str]:
    return [
        "id",
        "video_key",
        "frame",
        "fighter",
        "punch_type",
        "hand",
        "target",
        "effectiveness",
        "p_keep",
        "drop",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
