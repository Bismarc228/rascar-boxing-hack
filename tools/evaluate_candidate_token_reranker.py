#!/usr/bin/env python3
"""Evaluate sparse candidate-token component rerankers on fixed OOF rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, apply_temporal_context, score_pose_tracks
from tools.evaluate_exchange_state_gate import (
    apply_keep_mask,
    build_labels,
    fight_group,
    index_candidates,
    parse_floats,
    parse_ints,
    row_features,
)
from tools.evaluate_fixed_row_audio_gate import combined_features, compute_audio_tracks
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_rgb_fixed_row_attribute_model import (
    load_rgb_features,
    nearest_candidate_index,
    reconstruct_feature_index,
)


ATTR_VALUES = {
    "fighter": ["red", "blue"],
    "hand": ["left", "right"],
    "target": ["head", "body"],
    "punch_type": ["jab", "cross", "hook", "uppercut"],
    "effectiveness": ["landed", "blocked", "miss"],
}


@dataclass
class SequencePack:
    key: str
    group: str
    row_indices: list[int]
    x: np.ndarray
    y_clear: np.ndarray
    weights: np.ndarray
    y_class: np.ndarray
    class_mask: np.ndarray


class CandidateTokenModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.clear_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        z = self.input(x)
        z = self.encoder(z, src_key_padding_mask=pad_mask)
        return self.clear_head(z).squeeze(-1)


class CandidateTokenClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        z = self.input(x)
        z = self.encoder(z, src_key_padding_mask=pad_mask)
        return self.head(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--rgb-feature-cache", type=Path)
    parser.add_argument("--feature-mode", choices=["pose", "pose_audio"], default="pose_audio")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--context-feature", default="same_sum")
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=-0.2)
    parser.add_argument("--match-window", type=int, default=4)
    parser.add_argument("--pose-feature-windows", default="4,8,16")
    parser.add_argument("--audio-feature-windows", default="0,3,6,12,24")
    parser.add_argument("--audio-offsets", default="-12,-6,-3,0,3,6,12")
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=1800)
    parser.add_argument("--candidate-match-window", type=int, default=6)
    parser.add_argument("--heads", default="clear", help="One head for this smoke: clear, effectiveness, or punch_type.")
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--attribute-margins", default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--soft-time-only", type=float, default=0.40)
    parser.add_argument("--fp-weight", type=float, default=1.15)
    parser.add_argument("--time-only-weight", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-threshold", type=float)
    parser.add_argument("--write-margin", type=float)
    parser.add_argument("--write-diagnostics", type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    heads = [item.strip() for item in args.heads.split(",") if item.strip()]
    if len(heads) != 1 or heads[0] not in {"clear", "effectiveness", "punch_type"}:
        raise NotImplementedError("Use one --heads value: clear, effectiveness, or punch_type")
    head = heads[0]
    seed_everything(args.seed)
    device = pick_device(args.device)

    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, pred_rows)
    print_component_score("baseline", baseline, baseline, len(pred_rows), 0, 0, {})

    _labels_y, label_names = build_labels(pred_rows, gt_rows)
    attr_labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    sequences = build_sequences(args, pred_rows, video_by_key, keys, label_names, attr_labels, head)

    if head != "clear":
        probabilities = fit_oof_attribute_model(args, sequences, device, head)
        evaluate_attribute_head(args, head, pred_rows, gt_rows, baseline, probabilities)
        return 0

    p_keep = fit_oof_clear_model(args, sequences, device)

    print_label_summary(label_names, p_keep)
    oracle_rows = apply_keep_mask(pred_rows, np.asarray([label != "fp" for label in label_names], dtype=bool))
    oracle_score = score_predictions(gt_rows, oracle_rows)
    print_component_score(
        "oracle_drop_fp",
        oracle_score,
        baseline,
        len(oracle_rows),
        video_wins(oracle_score, baseline),
        len(pred_rows) - len(oracle_rows),
        drop_counts(label_names, np.asarray([label == "fp" for label in label_names], dtype=bool)),
    )

    results = []
    for threshold in parse_floats(args.thresholds):
        keep_mask = p_keep >= threshold
        rows = apply_keep_mask(pred_rows, keep_mask)
        score = score_predictions(gt_rows, rows)
        results.append(
            {
                "threshold": threshold,
                "score": score,
                "rows": rows,
                "keep_mask": keep_mask,
                "wins": video_wins(score, baseline),
            }
        )

    print(
        "threshold,macro,delta,time,fighter,punch_type,effectiveness,hand,target,"
        "fp_penalty,n_pred,n_tp,n_fp,n_fn,wins,dropped,"
        "dropped_fp,dropped_tp_scorable,dropped_tp_time_only,gate_clear_fp"
    )
    for item in sorted(results, key=lambda row: row["score"]["macro_score"], reverse=True)[: args.top_k]:
        score = item["score"]
        keep_mask = item["keep_mask"]
        counts = drop_counts(label_names, ~keep_mask)
        summary = score_summary(score)
        print(
            f"{item['threshold']:.6g},{score['macro_score']:.6f},"
            f"{score['macro_score'] - baseline['macro_score']:+.6f},"
            f"{summary['time']:.6f},{summary['fighter']:.6f},"
            f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
            f"{summary['hand']:.6f},{summary['target']:.6f},{summary['fp_penalty']:.6f},"
            f"{summary['n_pred']},{summary['n_tp']},{summary['n_fp']},{summary['n_fn']},"
            f"{item['wins']},{int((~keep_mask).sum())},"
            f"{counts.get('fp', 0)},{counts.get('tp_scorable', 0)},{counts.get('tp_time_only', 0)},"
            f"{int(clear_fp_gate(score, baseline, summary))}",
            flush=True,
        )

    if args.write_oof_rows:
        threshold = args.write_threshold
        if threshold is None:
            threshold = max(results, key=lambda row: row["score"]["macro_score"])["threshold"]
        rows = apply_keep_mask(pred_rows, p_keep >= threshold)
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} threshold={threshold} n_rows={len(rows)}")

    if args.write_diagnostics:
        write_diagnostics(args.write_diagnostics, pred_rows, label_names, p_keep)
        print(f"wrote_diagnostics={args.write_diagnostics}")
    return 0


def build_sequences(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    keys: list[str],
    label_names: list[str],
    attr_labels: list[dict[str, str] | None],
    head: str,
) -> list[SequencePack]:
    config = PoseHeuristicConfig(
        min_score=0.0,
        context_feature=args.context_feature,
        context_window=args.context_window,
        context_alpha=args.context_alpha,
    )
    candidates_by_frame = {}
    for key in keys:
        candidates_by_frame[key] = index_candidates(
            apply_temporal_context(score_pose_tracks(args.tracks_dir / f"{key}.jsonl", config), config)
        )

    rgb_by_row = None
    if args.rgb_feature_cache:
        rgb_by_row = build_rgb_features(args, pred_rows, keys)

    audio_by_key = {}
    if args.feature_mode == "pose_audio":
        for key in keys:
            video = video_by_key[key]
            audio_by_key[key] = compute_audio_tracks(
                args.data_root / video["video_path"],
                args.sample_rate,
                as_int(video["frame_count"], "frame_count"),
            )

    pose_windows = parse_ints(args.pose_feature_windows)
    audio_windows = parse_ints(args.audio_feature_windows)
    audio_offsets = parse_ints(args.audio_offsets)
    row_base = []
    for index, row in enumerate(pred_rows):
        key = row["video_key"]
        if args.feature_mode == "pose_audio":
            features = combined_features(
                row,
                video_by_key[key],
                candidates_by_frame[key],
                audio_by_key[key],
                args.match_window,
                pose_windows,
                audio_windows,
                audio_offsets,
                args.feature_mode,
            )
        else:
            features = row_features(
                row,
                video_by_key[key],
                candidates_by_frame[key],
                args.match_window,
                pose_windows,
            )
        extra = row_context_features(row, video_by_key[key])
        if rgb_by_row is not None:
            extra = np.concatenate([extra, rgb_by_row[index]])
        row_base.append(np.concatenate([features, extra]).astype(np.float32))

    row_indices_by_key: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        row_indices_by_key[row["video_key"]].append(index)

    sequences = []
    for key in keys:
        indices = sorted(row_indices_by_key[key], key=lambda idx: (as_int(pred_rows[idx]["frame"], "frame"), idx))
        x = np.stack([row_base[idx] for idx in indices])
        x = np.concatenate([x, sequence_context_features([pred_rows[idx] for idx in indices], video_by_key[key])], axis=1)
        y = np.asarray([label_target(label_names[idx], args.soft_time_only) for idx in indices], dtype=np.float32)
        weights = np.asarray([label_weight(label_names[idx], args) for idx in indices], dtype=np.float32)
        y_class, class_mask = class_targets(indices, attr_labels, head)
        sequences.append(
            SequencePack(
                key=key,
                group=fight_group(video_by_key[key]),
                row_indices=indices,
                x=x,
                y_clear=y,
                weights=weights,
                y_class=y_class,
                class_mask=class_mask,
            )
        )
    return sequences


def build_rgb_features(args: argparse.Namespace, pred_rows: list[dict[str, str]], keys: list[str]) -> np.ndarray:
    features = load_rgb_features(args.rgb_feature_cache)
    candidates_by_key, feature_by_key = reconstruct_feature_index(args, keys, features)
    dim = features.shape[-1] * 2
    output = []
    for row in pred_rows:
        key = row["video_key"]
        index = nearest_candidate_index(row, candidates_by_key[key], args.candidate_match_window)
        if index is None:
            output.append(np.zeros(dim, dtype=np.float32))
            continue
        clip = feature_by_key[key][index]
        output.append(np.concatenate([clip.mean(axis=0), clip.std(axis=0)]).astype(np.float32))
    return np.stack(output)


def row_context_features(row: dict[str, str], video: dict[str, str]) -> np.ndarray:
    values = [
        as_int(row["frame"], "frame") / max(1.0, float(video["frame_count"])),
        np.log1p(as_int(row["frame"], "frame")),
    ]
    for column, choices in ATTR_VALUES.items():
        for value in choices:
            values.append(float(row.get(column) == value))
    return np.asarray(values, dtype=np.float32)


def sequence_context_features(rows: list[dict[str, str]], video: dict[str, str]) -> np.ndarray:
    frames = np.asarray([as_int(row["frame"], "frame") for row in rows], dtype=np.float32)
    n = len(rows)
    output = []
    for index, row in enumerate(rows):
        frame = frames[index]
        prev_gap = frame - frames[index - 1] if index > 0 else float(video["frame_count"])
        next_gap = frames[index + 1] - frame if index + 1 < n else float(video["frame_count"])
        same_stream_gaps = [
            abs(frame - frames[j])
            for j, other in enumerate(rows)
            if j != index and other["fighter"] == row["fighter"] and other["hand"] == row["hand"]
        ]
        same_fighter_gaps = [
            abs(frame - frames[j])
            for j, other in enumerate(rows)
            if j != index and other["fighter"] == row["fighter"]
        ]
        local_12 = float(np.sum(np.abs(frames - frame) <= 12) - 1)
        local_24 = float(np.sum(np.abs(frames - frame) <= 24) - 1)
        output.append(
            [
                np.log1p(max(0.0, prev_gap)),
                np.log1p(max(0.0, next_gap)),
                np.log1p(min(same_stream_gaps) if same_stream_gaps else float(video["frame_count"])),
                np.log1p(min(same_fighter_gaps) if same_fighter_gaps else float(video["frame_count"])),
                local_12,
                local_24,
                index / max(1.0, n - 1),
                n / max(1.0, float(video["frame_count"])),
            ]
        )
    return np.asarray(output, dtype=np.float32)


def fit_oof_clear_model(args: argparse.Namespace, sequences: list[SequencePack], device: torch.device) -> np.ndarray:
    output = np.zeros(sum(len(seq.row_indices) for seq in sequences), dtype=np.float32)
    groups = sorted({seq.group for seq in sequences})
    for group in groups:
        train_sequences = [seq for seq in sequences if seq.group != group]
        valid_sequences = [seq for seq in sequences if seq.group == group]
        scaler = fit_scaler(np.concatenate([seq.x for seq in train_sequences], axis=0))
        train_scaled = [replace_x(seq, apply_scaler(seq.x, scaler)) for seq in train_sequences]
        valid_scaled = [replace_x(seq, apply_scaler(seq.x, scaler)) for seq in valid_sequences]
        model = train_clear_model(args, train_scaled, device)
        model.eval()
        with torch.no_grad():
            for seq in valid_scaled:
                x = torch.from_numpy(seq.x).unsqueeze(0).to(device)
                pad = torch.zeros((1, len(seq.row_indices)), dtype=torch.bool, device=device)
                prob = torch.sigmoid(model(x, pad)).squeeze(0).detach().cpu().numpy().astype(np.float32)
                for row_index, value in zip(seq.row_indices, prob):
                    output[row_index] = float(value)
        train_rows = sum(len(seq.row_indices) for seq in train_sequences)
        valid_rows = sum(len(seq.row_indices) for seq in valid_sequences)
        print(f"fold={group} train_rows={train_rows} valid_rows={valid_rows}", flush=True)
    return output


def fit_oof_attribute_model(
    args: argparse.Namespace,
    sequences: list[SequencePack],
    device: torch.device,
    head: str,
) -> np.ndarray:
    n_rows = sum(len(seq.row_indices) for seq in sequences)
    n_classes = len(ATTR_VALUES[head])
    output = np.zeros((n_rows, n_classes), dtype=np.float32)
    groups = sorted({seq.group for seq in sequences})
    for group in groups:
        train_sequences = [seq for seq in sequences if seq.group != group]
        valid_sequences = [seq for seq in sequences if seq.group == group]
        scaler = fit_scaler(np.concatenate([seq.x for seq in train_sequences], axis=0))
        train_scaled = [replace_x(seq, apply_scaler(seq.x, scaler)) for seq in train_sequences]
        valid_scaled = [replace_x(seq, apply_scaler(seq.x, scaler)) for seq in valid_sequences]
        model = train_attribute_model(args, train_scaled, device, n_classes)
        model.eval()
        with torch.no_grad():
            for seq in valid_scaled:
                x = torch.from_numpy(seq.x).unsqueeze(0).to(device)
                pad = torch.zeros((1, len(seq.row_indices)), dtype=torch.bool, device=device)
                prob = torch.softmax(model(x, pad), dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
                for row_index, row_prob in zip(seq.row_indices, prob):
                    output[row_index] = row_prob
        train_rows = int(sum(seq.class_mask.sum() for seq in train_sequences))
        valid_rows = int(sum(seq.class_mask.sum() for seq in valid_sequences))
        print(f"fold={group} train_labeled={train_rows} valid_labeled={valid_rows}", flush=True)
    return output


def train_clear_model(args: argparse.Namespace, sequences: list[SequencePack], device: torch.device) -> CandidateTokenModel:
    input_dim = sequences[0].x.shape[1]
    model = CandidateTokenModel(input_dim, args.hidden, args.layers, args.attention_heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _epoch in range(args.epochs):
        random.shuffle(sequences)
        model.train()
        for batch in make_batches(sequences, batch_size=min(4, max(1, len(sequences)))):
            x, y, weights, pad = tensors_for_batch(batch, device)
            logits = model(x, pad)
            loss_raw = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (loss_raw * weights * (~pad).float()).sum() / torch.clamp((weights * (~pad).float()).sum(), min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


def train_attribute_model(
    args: argparse.Namespace,
    sequences: list[SequencePack],
    device: torch.device,
    n_classes: int,
) -> CandidateTokenClassifier:
    input_dim = sequences[0].x.shape[1]
    model = CandidateTokenClassifier(
        input_dim,
        args.hidden,
        args.layers,
        args.attention_heads,
        args.dropout,
        n_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = torch.from_numpy(class_weight_vector(sequences, n_classes)).to(device)
    for _epoch in range(args.epochs):
        random.shuffle(sequences)
        model.train()
        for batch in make_batches(sequences, batch_size=min(4, max(1, len(sequences)))):
            x, y, _weights, pad, class_mask = tensors_for_class_batch(batch, device)
            logits = model(x, pad)
            loss_raw = nn.functional.cross_entropy(
                logits.reshape(-1, n_classes),
                y.reshape(-1),
                weight=class_weights,
                reduction="none",
            ).reshape_as(y)
            mask = class_mask & (~pad)
            loss = (loss_raw * mask.float()).sum() / torch.clamp(mask.float().sum(), min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


def make_batches(sequences: list[SequencePack], batch_size: int) -> list[list[SequencePack]]:
    return [sequences[start : start + batch_size] for start in range(0, len(sequences), batch_size)]


def tensors_for_batch(batch: list[SequencePack], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(seq.row_indices) for seq in batch)
    dim = batch[0].x.shape[1]
    x = np.zeros((len(batch), max_len, dim), dtype=np.float32)
    y = np.zeros((len(batch), max_len), dtype=np.float32)
    weights = np.zeros((len(batch), max_len), dtype=np.float32)
    pad = np.ones((len(batch), max_len), dtype=bool)
    for i, seq in enumerate(batch):
        n = len(seq.row_indices)
        x[i, :n] = seq.x
        y[i, :n] = seq.y_clear
        weights[i, :n] = seq.weights
        pad[i, :n] = False
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(weights).to(device),
        torch.from_numpy(pad).to(device),
    )


def tensors_for_class_batch(
    batch: list[SequencePack],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(seq.row_indices) for seq in batch)
    dim = batch[0].x.shape[1]
    x = np.zeros((len(batch), max_len, dim), dtype=np.float32)
    y = np.zeros((len(batch), max_len), dtype=np.int64)
    weights = np.zeros((len(batch), max_len), dtype=np.float32)
    pad = np.ones((len(batch), max_len), dtype=bool)
    class_mask = np.zeros((len(batch), max_len), dtype=bool)
    for i, seq in enumerate(batch):
        n = len(seq.row_indices)
        x[i, :n] = seq.x
        y[i, :n] = seq.y_class
        weights[i, :n] = 1.0
        pad[i, :n] = False
        class_mask[i, :n] = seq.class_mask
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(weights).to(device),
        torch.from_numpy(pad).to(device),
        torch.from_numpy(class_mask).to(device),
    )


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(x: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = scaler
    out = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    return np.clip(out, -8.0, 8.0).astype(np.float32)


def replace_x(seq: SequencePack, x: np.ndarray) -> SequencePack:
    return SequencePack(seq.key, seq.group, seq.row_indices, x, seq.y_clear, seq.weights, seq.y_class, seq.class_mask)


def class_targets(
    indices: list[int],
    attr_labels: list[dict[str, str] | None],
    head: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(len(indices), dtype=np.int64)
    mask = np.zeros(len(indices), dtype=bool)
    if head == "clear":
        return y, mask
    values = ATTR_VALUES[head]
    value_to_index = {value: index for index, value in enumerate(values)}
    for out_index, row_index in enumerate(indices):
        label = attr_labels[row_index]
        if label is None:
            continue
        value = label.get(head, "")
        if value not in value_to_index:
            continue
        y[out_index] = value_to_index[value]
        mask[out_index] = True
    return y, mask


def class_weight_vector(sequences: list[SequencePack], n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.float32)
    for seq in sequences:
        for value in seq.y_class[seq.class_mask]:
            counts[int(value)] += 1.0
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return np.clip(weights, 0.25, 5.0).astype(np.float32)


def matched_attribute_labels(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    label_window: int,
) -> list[dict[str, str] | None]:
    labels: list[dict[str, str] | None] = [None] * len(pred_rows)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    global_indices_by_key: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        global_indices_by_key[row["video_key"]].append(index)
    for key, rows in pred_by_key.items():
        gt_video = gt_by_key.get(key, [])
        for match in match_events(gt_video, rows):
            pred = rows[match.pred_index]
            gt = gt_video[match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) > label_window:
                continue
            labels[global_indices_by_key[key][match.pred_index]] = gt
    return labels


def evaluate_attribute_head(
    args: argparse.Namespace,
    head: str,
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    baseline: dict[str, Any],
    probabilities: np.ndarray,
) -> None:
    values = ATTR_VALUES[head]
    print_attribute_probability_summary(head, pred_rows, gt_rows, probabilities)
    print(
        "margin,macro,delta,time,fighter,punch_type,effectiveness,hand,target,"
        "fp_penalty,n_pred,n_tp,n_fp,n_fn,wins,changed,gate_component"
    )
    rows_by_margin = {}
    for margin in parse_floats(args.attribute_margins):
        rows, changed = apply_attribute_margin(pred_rows, head, values, probabilities, margin)
        rows_by_margin[margin] = rows
        score = score_predictions(gt_rows, rows)
        summary = score_summary(score)
        print(
            f"{margin:.6g},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:+.6f},"
            f"{summary['time']:.6f},{summary['fighter']:.6f},"
            f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
            f"{summary['hand']:.6f},{summary['target']:.6f},{summary['fp_penalty']:.6f},"
            f"{summary['n_pred']},{summary['n_tp']},{summary['n_fp']},{summary['n_fn']},"
            f"{video_wins(score, baseline)},{changed},{int(attribute_gate(head, score, baseline, summary))}",
            flush=True,
        )
    if args.write_oof_rows:
        margin = args.write_margin
        if margin is None:
            best_margin, _best_rows = max(
                rows_by_margin.items(),
                key=lambda item: score_predictions(gt_rows, item[1])["macro_score"],
            )
            margin = best_margin
        rows = rows_by_margin[margin]
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} margin={margin} n_rows={len(rows)}")
    if args.write_diagnostics:
        write_attribute_diagnostics(args.write_diagnostics, pred_rows, head, values, probabilities)
        print(f"wrote_diagnostics={args.write_diagnostics}")


def apply_attribute_margin(
    rows: list[dict[str, str]],
    head: str,
    values: list[str],
    probabilities: np.ndarray,
    margin: float,
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for row, prob in zip(rows, probabilities):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        pred_index = int(np.argmax(prob))
        pred_value = values[pred_index]
        current_value = row[head]
        current_prob = float(prob[values.index(current_value)]) if current_value in values else 0.0
        if pred_value != current_value and float(prob[pred_index]) - current_prob >= margin:
            item[head] = pred_value
            changed += 1
        output.append(item)
    return output, changed


def attribute_gate(head: str, score: dict[str, Any], baseline: dict[str, Any], summary: dict[str, float | int]) -> bool:
    base = score_summary(baseline)
    if head == "effectiveness":
        return bool(
            float(summary["effectiveness"]) >= max(float(base["effectiveness"]) + 0.020, 0.300)
            and score["macro_score"] >= max(baseline["macro_score"] + 0.001, 0.409)
        )
    if head == "punch_type":
        return bool(
            float(summary["punch_type"]) >= max(float(base["punch_type"]) + 0.030, 0.235)
            and score["macro_score"] >= baseline["macro_score"] + 0.003
        )
    return False


def print_attribute_probability_summary(
    head: str,
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    probabilities: np.ndarray,
) -> None:
    labels = matched_attribute_labels(pred_rows, gt_rows, 12)
    values = ATTR_VALUES[head]
    counts = Counter(label[head] for label in labels if label is not None and label.get(head) in values)
    pred_counts = Counter(values[int(np.argmax(row))] for row in probabilities)
    print("labels_" + head + "=" + ",".join(f"{key}:{counts[key]}" for key in values), flush=True)
    print("pred_" + head + "=" + ",".join(f"{key}:{pred_counts[key]}" for key in values), flush=True)


def write_attribute_diagnostics(
    path: Path,
    rows: list[dict[str, str]],
    head: str,
    values: list[str],
    probabilities: np.ndarray,
) -> None:
    output = []
    for row, prob in zip(rows, probabilities):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        item["pred_" + head] = values[int(np.argmax(prob))]
        for value, score in zip(values, prob):
            item[f"p_{head}_{value}"] = f"{float(score):.9f}"
        output.append(item)
    fields = SUBMISSION_COLUMNS + ["pred_" + head] + [f"p_{head}_{value}" for value in values]
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def label_target(label: str, soft_time_only: float) -> float:
    if label == "tp_scorable":
        return 1.0
    if label == "tp_time_only":
        return soft_time_only
    return 0.0


def label_weight(label: str, args: argparse.Namespace) -> float:
    if label == "fp":
        return args.fp_weight
    if label == "tp_time_only":
        return args.time_only_weight
    return 1.0


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_summary(score: dict[str, Any]) -> dict[str, float | int]:
    by_video = score["by_video"]
    values = list(by_video.values())
    return {
        "time": float(np.mean([item["score_time"] for item in values])),
        "fighter": float(np.mean([item["score_fighter"] for item in values])),
        "punch_type": float(np.mean([item["score_punch_type"] for item in values])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in values])),
        "hand": float(np.mean([item["score_hand"] for item in values])),
        "target": float(np.mean([item["score_target"] for item in values])),
        "fp_penalty": float(np.mean([item["fp_penalty"] for item in values])),
        "n_pred": int(sum(item["n_pred"] for item in values)),
        "n_tp": int(sum(item["n_tp"] for item in values)),
        "n_fp": int(sum(item["n_fp"] for item in values)),
        "n_fn": int(sum(item["n_fn"] for item in values)),
    }


def print_component_score(
    label: str,
    score: dict[str, Any],
    baseline: dict[str, Any],
    n_rows: int,
    wins: int,
    dropped: int,
    dropped_counts: dict[str, int],
) -> None:
    summary = score_summary(score)
    print(
        f"{label}: macro={score['macro_score']:.6f},delta={score['macro_score'] - baseline['macro_score']:+.6f},"
        f"time={summary['time']:.6f},fighter={summary['fighter']:.6f},"
        f"punch_type={summary['punch_type']:.6f},effectiveness={summary['effectiveness']:.6f},"
        f"hand={summary['hand']:.6f},target={summary['target']:.6f},"
        f"fp_penalty={summary['fp_penalty']:.6f},n_pred={summary['n_pred']},"
        f"n_tp={summary['n_tp']},n_fp={summary['n_fp']},n_fn={summary['n_fn']},"
        f"wins={wins},n_rows={n_rows},dropped={dropped},"
        f"dropped_fp={dropped_counts.get('fp', 0)},"
        f"dropped_tp_scorable={dropped_counts.get('tp_scorable', 0)},"
        f"dropped_tp_time_only={dropped_counts.get('tp_time_only', 0)}",
        flush=True,
    )


def clear_fp_gate(score: dict[str, Any], baseline: dict[str, Any], summary: dict[str, float | int]) -> bool:
    base = score_summary(baseline)
    return bool(
        score["macro_score"] - baseline["macro_score"] >= 0.003
        and float(summary["fp_penalty"]) <= max(0.0, float(base["fp_penalty"]) - 0.004)
        and int(summary["n_fp"]) <= int(base["n_fp"]) - 10
        and int(summary["n_tp"]) >= int(base["n_tp"]) - 8
        and float(summary["time"]) >= float(base["time"]) - 0.003
        and float(summary["fighter"]) >= float(base["fighter"]) - 0.003
    )


def drop_counts(labels: list[str], drop_mask: np.ndarray) -> dict[str, int]:
    counts = Counter(label for label, drop in zip(labels, drop_mask) if drop)
    return dict(counts)


def print_label_summary(labels: list[str], p_keep: np.ndarray) -> None:
    counts = Counter(labels)
    print("labels=" + ",".join(f"{name}:{counts[name]}" for name in ["tp_scorable", "tp_time_only", "fp"]))
    for label in ["tp_scorable", "tp_time_only", "fp"]:
        values = p_keep[np.asarray([item == label for item in labels], dtype=bool)]
        if len(values) == 0:
            continue
        print(
            f"p_keep_{label}=n:{len(values)},mean:{float(values.mean()):.4f},"
            f"p10:{float(np.quantile(values, 0.1)):.4f},"
            f"p50:{float(np.quantile(values, 0.5)):.4f},"
            f"p90:{float(np.quantile(values, 0.9)):.4f}",
            flush=True,
        )


def write_diagnostics(path: Path, rows: list[dict[str, str]], labels: list[str], p_keep: np.ndarray) -> None:
    output = []
    for row, label, prob in zip(rows, labels, p_keep):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        item["label"] = label
        item["p_keep"] = f"{float(prob):.9f}"
        output.append(item)
    fields = SUBMISSION_COLUMNS + ["label", "p_keep"]
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    raise SystemExit(main())
