#!/usr/bin/env python3
"""Train OOF neural attribute heads on cached ViT/VideoMAE features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_vit_fixed_row_attribute_head import (
    ATTR_COLUMNS,
    apply_combo,
    apply_margin,
    component_summary,
    fight_group,
    matched_attribute_labels,
    parse_combos,
    parse_floats,
    parse_list,
    print_head_diagnostics,
    print_label_counts,
    print_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-caches", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--feature-pooling",
        choices=["center_mean_std", "center", "mean", "center_mean", "mean_std"],
        default="center_mean_std",
    )
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--heads", default="punch_type,effectiveness")
    parser.add_argument("--margins", default="0.0,0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--combos", default="")
    parser.add_argument("--pca-components", type=int, default=160, help="set <=0 to train on scaled pooled features without PCA")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--mlp-depth", type=int, default=1)
    parser.add_argument("--residual-head", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.18)
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-head", choices=ATTR_COLUMNS)
    parser.add_argument("--write-margin", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mlp_depth < 1:
        raise ValueError("--mlp-depth must be >= 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    keys = {row["video_key"] for row in pred_rows}
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in keys
    ]
    x = load_feature_caches(args.feature_caches, args.feature_pooling)
    if len(x) != len(pred_rows):
        raise ValueError(f"feature rows {len(x)} != prediction rows {len(pred_rows)}")
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])

    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, baseline, len(pred_rows), 0)
    heads = parse_list(args.heads)
    predictions, probabilities = oof_predictions(args, x, labels, groups, pred_rows, heads)
    print_head_diagnostics(labels, pred_rows, predictions, heads)

    rows_by_variant = {}
    print("variant,macro,delta,time,fighter,punch_type,effectiveness,hand,target,fp_penalty,wins,changed")
    for head in heads:
        for margin in parse_floats(args.margins):
            variant = f"{head}_margin_{margin:g}"
            rows, changed = apply_margin(pred_rows, predictions, probabilities, head, margin)
            rows_by_variant[(head, margin)] = rows
            score = score_predictions(gt_rows, rows)
            summary = component_summary(score)
            print(
                f"{variant},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:+.6f},"
                f"{summary['time']:.6f},{summary['fighter']:.6f},{summary['punch_type']:.6f},"
                f"{summary['effectiveness']:.6f},{summary['hand']:.6f},{summary['target']:.6f},"
                f"{summary['fp_penalty']:.6f},{video_wins(score, baseline)},{changed}",
                flush=True,
            )
    for combo in parse_combos(args.combos):
        variant = "combo_" + "_".join(f"{head}{margin:g}" for head, margin in combo)
        rows, changed = apply_combo(pred_rows, predictions, probabilities, combo)
        score = score_predictions(gt_rows, rows)
        summary = component_summary(score)
        print(
            f"{variant},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:+.6f},"
            f"{summary['time']:.6f},{summary['fighter']:.6f},{summary['punch_type']:.6f},"
            f"{summary['effectiveness']:.6f},{summary['hand']:.6f},{summary['target']:.6f},"
            f"{summary['fp_penalty']:.6f},{video_wins(score, baseline)},{changed}",
            flush=True,
        )
    print_label_counts(labels)
    if args.write_oof_rows:
        if args.write_head is None or args.write_margin is None:
            raise ValueError("--write-oof-rows requires --write-head and --write-margin")
        rows = rows_by_variant[(args.write_head, args.write_margin)]
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} head={args.write_head} margin={args.write_margin}")
    return 0


def load_feature_caches(paths: list[Path], pooling: str = "center_mean_std") -> np.ndarray:
    arrays = []
    for path in paths:
        data = np.load(path)
        features = data["features"].astype(np.float32)
        if features.ndim != 3:
            raise ValueError(f"expected [N,T,D] in {path}, got {features.shape}")
        mean = features.mean(axis=1)
        std = features.std(axis=1)
        center = features[:, features.shape[1] // 2]
        if pooling == "center_mean_std":
            pooled = [center, mean, std]
        elif pooling == "center":
            pooled = [center]
        elif pooling == "mean":
            pooled = [mean]
        elif pooling == "center_mean":
            pooled = [center, mean]
        elif pooling == "mean_std":
            pooled = [mean, std]
        else:
            raise ValueError(f"unknown feature pooling {pooling}")
        arrays.append(np.concatenate(pooled, axis=1).astype(np.float32))
    rows = {array.shape[0] for array in arrays}
    if len(rows) != 1:
        raise ValueError(f"feature caches have different row counts: {sorted(rows)}")
    return np.concatenate(arrays, axis=1).astype(np.float32)


def oof_predictions(
    args: argparse.Namespace,
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    groups: np.ndarray,
    pred_rows: list[dict[str, str]],
    heads: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, float]]]]:
    output = {head: np.asarray([row[head] for row in pred_rows], dtype=object) for head in ATTR_COLUMNS}
    probabilities = {head: [{row[head]: 1.0} for row in pred_rows] for head in ATTR_COLUMNS}
    trainable = np.asarray([label is not None for label in labels], dtype=bool)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    device = torch.device(args.device)
    if device.type == "cuda":
        visible = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
        if visible != ["1"]:
            raise RuntimeError("CUDA neural head training must run with CUDA_VISIBLE_DEVICES=1")
        device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        f"training_device={device} device_name={device_name} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '') or '<unset>'}",
        flush=True,
    )
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = (groups != group) & trainable
        if train.sum() < 20:
            print(f"fold={group} train={int(train.sum())} skipped", flush=True)
            continue
        for head in heads:
            encoder = LabelEncoder()
            train_indices = np.where(train)[0]
            values = [labels[index][head] for index in train_indices]  # type: ignore[index]
            y = encoder.fit_transform(values).astype(np.int64)
            if len(encoder.classes_) == 1:
                continue
            fold_probs = []
            for seed in seeds:
                fold_probs.append(train_one_seed(args, x, y, train_indices, valid, encoder, seed, device))
            proba = np.mean(fold_probs, axis=0)
            pred_encoded = proba.argmax(axis=1)
            valid_indices = np.where(valid)[0]
            output[head][valid] = encoder.inverse_transform(pred_encoded)
            class_values = [str(value) for value in encoder.classes_]
            for index, row_probs in zip(valid_indices, proba):
                probabilities[head][index] = {
                    class_value: float(prob)
                    for class_value, prob in zip(class_values, row_probs)
                }
        print(f"fold={group} train={int(train.sum())} valid={int(valid.sum())}", flush=True)
    return output, probabilities


def train_one_seed(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_mask: np.ndarray,
    encoder: LabelEncoder,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    set_seed(seed)
    fit_idx, val_idx = stratified_holdout(y, args.val_frac, seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_indices]).astype(np.float32)
    x_valid = scaler.transform(x[valid_mask]).astype(np.float32)
    if args.pca_components > 0:
        n_components = min(args.pca_components, x_train.shape[0] - 1, x_train.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        x_train = pca.fit_transform(x_train).astype(np.float32)
        x_valid = pca.transform(x_valid).astype(np.float32)
        in_dim = n_components
    else:
        in_dim = x_train.shape[1]

    model = MLPHead(
        in_dim,
        args.hidden_dim,
        len(encoder.classes_),
        args.dropout,
        args.mlp_depth,
        args.residual_head,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(y[fit_idx], len(encoder.classes_), args.class_weight_power).to(device)
    fit_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train[fit_idx]), torch.from_numpy(y[fit_idx])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x = torch.from_numpy(x_train[val_idx]).to(device)
    val_y = torch.from_numpy(y[val_idx]).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(args.epochs):
        model.train()
        for xb, yb in fit_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(args, logits, yb, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(args, model(val_x), val_y, weights).detach().cpu())
        if val_loss + 1e-5 < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(x_valid).to(device))
        if args.temperature != 1.0:
            logits = logits / args.temperature
        return torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        depth: int,
        residual: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            if residual:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.extend(
                    [
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def loss_fn(args: argparse.Namespace, logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(
        logits,
        target,
        weight=weights,
        label_smoothing=args.label_smoothing if args.loss == "ce" else 0.0,
        reduction="none",
    )
    if args.loss == "focal":
        pt = torch.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1).clamp_min(1e-6)
        ce = ce * (1.0 - pt).pow(args.focal_gamma)
    return ce.mean()


def class_weights(y: np.ndarray, n_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = (len(y) / (n_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def stratified_holdout(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit = []
    val = []
    for cls in sorted(set(y.tolist())):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_val = max(1, int(round(len(cls_indices) * frac))) if len(cls_indices) > 2 else 1
        val.extend(cls_indices[:n_val].tolist())
        fit.extend(cls_indices[n_val:].tolist())
    return np.asarray(fit, dtype=np.int64), np.asarray(val, dtype=np.int64)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
