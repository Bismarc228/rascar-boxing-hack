#!/usr/bin/env python3
"""Tune an OOF effectiveness head on cached VideoMAE features."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import read_csv_rows, write_csv_rows
from rascar_boxing.metric import score_predictions
from tools.evaluate_pose_selection_variants import video_wins
from tools.evaluate_vit_fixed_row_attribute_head import (
    apply_margin,
    component_summary,
    fight_group,
    matched_attribute_labels,
    parse_floats,
    print_label_counts,
    print_score,
)

HEAD = "effectiveness"
ATTR_COLUMNS = ["punch_type", "effectiveness", "hand", "target", "fighter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-caches", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--feature-pooling",
        choices=["center_mean_std", "center", "mean", "center_mean", "mean_std", "sequence", "flatten"],
        default="center_mean_std",
    )
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--margins", default="0.0,0.1,0.2,0.3,0.5")
    parser.add_argument("--pca-components", type=int, default=0, help="set <=0 to skip PCA")
    parser.add_argument("--head-architecture", choices=["mlp", "resnet_mlp"], default="mlp")
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
    parser.add_argument("--class-weight-power", type=float, default=0.7)
    parser.add_argument("--class-weight-mode", choices=["power", "effective"], default="power")
    parser.add_argument("--effective-beta", type=float, default=0.99)
    parser.add_argument("--class-weight-min", type=float, default=0.0)
    parser.add_argument("--class-weight-max", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=["val_loss", "balanced_acc", "macro_f1"], default="val_loss")
    parser.add_argument("--val-frac", type=float, default=0.18)
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-margin", type=float)
    parser.add_argument("--report-video-deltas", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
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
    predictions, probabilities = oof_predictions(args, x, labels, groups, pred_rows)
    print_head_diagnostics(labels, pred_rows, predictions)

    rows_by_margin = {}
    score_by_margin = {}
    print("variant,macro,delta,time,fighter,punch_type,effectiveness,hand,target,fp_penalty,wins,changed")
    for margin in parse_floats(args.margins):
        rows, changed = apply_margin(pred_rows, predictions, probabilities, HEAD, margin)
        score = score_predictions(gt_rows, rows)
        summary = component_summary(score)
        rows_by_margin[margin] = rows
        score_by_margin[margin] = score
        print(
            f"{HEAD}_margin_{margin:g},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:+.6f},"
            f"{summary['time']:.6f},{summary['fighter']:.6f},{summary['punch_type']:.6f},"
            f"{summary['effectiveness']:.6f},{summary['hand']:.6f},{summary['target']:.6f},"
            f"{summary['fp_penalty']:.6f},{video_wins(score, baseline)},{changed}",
            flush=True,
        )
    print_label_counts(labels)
    print_best_rows(score_by_margin, baseline)
    if args.report_video_deltas and score_by_margin:
        best_margin = max(score_by_margin, key=lambda margin: score_by_margin[margin]["macro_score"])
        print_video_deltas(best_margin, score_by_margin[best_margin], baseline)
    if args.write_oof_rows:
        if args.write_margin is None:
            raise ValueError("--write-oof-rows requires --write-margin")
        rows = rows_by_margin[args.write_margin]
        write_csv_rows(args.write_oof_rows, rows, SUBMISSION_COLUMNS)
        print(f"wrote_oof_rows={args.write_oof_rows} head={HEAD} margin={args.write_margin}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.mlp_depth < 1:
        raise ValueError("--mlp-depth must be >= 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.effective_beta <= 0 or args.effective_beta >= 1:
        raise ValueError("--effective-beta must be in (0, 1)")
    if args.class_weight_min and args.class_weight_max and args.class_weight_min > args.class_weight_max:
        raise ValueError("--class-weight-min cannot exceed --class-weight-max")


def load_feature_caches(paths: list[Path], pooling: str) -> np.ndarray:
    arrays = []
    for path in paths:
        data = np.load(path)
        features = data["features"].astype(np.float32)
        if features.ndim != 3:
            raise ValueError(f"expected [N,T,D] in {path}, got {features.shape}")
        if pooling == "sequence":
            arrays.append(features)
            continue
        if pooling == "flatten":
            arrays.append(features.reshape(features.shape[0], features.shape[1] * features.shape[2]))
            continue
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
    row_counts = {array.shape[0] for array in arrays}
    if len(row_counts) != 1:
        raise ValueError(f"feature caches have different row counts: {sorted(row_counts)}")
    if pooling == "sequence":
        temporal_counts = {array.shape[1] for array in arrays}
        if len(temporal_counts) != 1:
            raise ValueError(f"sequence feature caches have different temporal counts: {sorted(temporal_counts)}")
        return np.concatenate(arrays, axis=2).astype(np.float32)
    return np.concatenate(arrays, axis=1).astype(np.float32)


def oof_predictions(
    args: argparse.Namespace,
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    groups: np.ndarray,
    pred_rows: list[dict[str, str]],
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
        encoder = LabelEncoder()
        train_indices = np.where(train)[0]
        values = [labels[index][HEAD] for index in train_indices]  # type: ignore[index]
        y = encoder.fit_transform(values).astype(np.int64)
        if len(encoder.classes_) == 1:
            continue
        fold_probs = []
        for seed in seeds:
            fold_probs.append(train_one_seed(args, x, y, train_indices, valid, encoder, seed, device))
        proba = np.mean(fold_probs, axis=0)
        pred_encoded = proba.argmax(axis=1)
        valid_indices = np.where(valid)[0]
        output[HEAD][valid] = encoder.inverse_transform(pred_encoded)
        class_values = [str(value) for value in encoder.classes_]
        for index, row_probs in zip(valid_indices, proba):
            probabilities[HEAD][index] = {
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
    x_train, x_valid, in_dim = transform_fold_features(args, x, train_indices, valid_mask, seed)
    x_fit = select_rows(x_train, fit_idx)
    x_val = select_rows(x_train, val_idx)
    if args.pca_components > 0:
        print(f"fold_pca_components={in_dim}", flush=True)

    model = build_head(args, in_dim, len(encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(
        y[fit_idx],
        len(encoder.classes_),
        args.class_weight_mode,
        args.class_weight_power,
        args.effective_beta,
        args.class_weight_min,
        args.class_weight_max,
    ).to(device)
    fit_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y[fit_idx])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y[val_idx]).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_value = float("inf") if args.selection_metric == "val_loss" else -float("inf")
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
            val_logits = model(val_x)
            metric_value = validation_metric(args, val_logits, val_y, weights)
        improved = (
            metric_value < best_value - 1e-5
            if args.selection_metric == "val_loss"
            else metric_value > best_value + 1e-5
        )
        if improved:
            best_value = metric_value
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


def transform_fold_features(
    args: argparse.Namespace,
    x: np.ndarray,
    train_indices: np.ndarray,
    valid_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if x.ndim == 2:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_indices]).astype(np.float32)
        x_valid = scaler.transform(x[valid_mask]).astype(np.float32)
        if args.pca_components > 0:
            n_components = min(args.pca_components, x_train.shape[0] - 1, x_train.shape[1])
            pca = PCA(n_components=n_components, random_state=seed)
            x_train = pca.fit_transform(x_train).astype(np.float32)
            x_valid = pca.transform(x_valid).astype(np.float32)
            return x_train, x_valid, n_components
        return x_train, x_valid, x_train.shape[1]
    if x.ndim == 3:
        train_tokens = x[train_indices].reshape(-1, x.shape[2])
        scaler = StandardScaler()
        scaler.fit(train_tokens)
        x_train = scaler.transform(x[train_indices].reshape(-1, x.shape[2])).reshape(
            len(train_indices), x.shape[1], x.shape[2]
        ).astype(np.float32)
        x_valid = scaler.transform(x[valid_mask].reshape(-1, x.shape[2])).reshape(
            int(valid_mask.sum()), x.shape[1], x.shape[2]
        ).astype(np.float32)
        if args.pca_components > 0:
            n_components = min(args.pca_components, train_tokens.shape[0] - 1, x.shape[2])
            pca = PCA(n_components=n_components, random_state=seed)
            pca.fit(scaler.transform(train_tokens))
            x_train = pca.transform(x_train.reshape(-1, x.shape[2])).reshape(
                len(train_indices), x.shape[1], n_components
            ).astype(np.float32)
            x_valid = pca.transform(x_valid.reshape(-1, x.shape[2])).reshape(
                int(valid_mask.sum()), x.shape[1], n_components
            ).astype(np.float32)
            return x_train, x_valid, n_components
        return x_train, x_valid, x_train.shape[2]
    raise ValueError(f"expected 2D or 3D features, got shape={x.shape}")


def select_rows(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return x[indices]


def build_head(args: argparse.Namespace, in_dim: int, out_dim: int) -> nn.Module:
    if args.head_architecture == "resnet_mlp":
        return ResNetMLPHead(in_dim, args.hidden_dim, out_dim, args.dropout, args.mlp_depth)
    return MLPHead(in_dim, args.hidden_dim, out_dim, args.dropout, args.mlp_depth, args.residual_head)


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


class ResNetMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, depth: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(max(1, depth))])
        self.temporal_blocks = nn.Sequential(
            *[TemporalResidualBlock(hidden_dim, dropout) for _ in range(max(1, depth))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        if z.ndim == 3:
            z = z.transpose(1, 2)
            z = self.temporal_blocks(z)
            z = z.mean(dim=2)
        else:
            z = self.blocks(z)
        return self.head(z)


class TemporalResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


def validation_metric(
    args: argparse.Namespace,
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    if args.selection_metric == "val_loss":
        return float(loss_fn(args, logits, target, weights).detach().cpu())
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    if args.selection_metric == "balanced_acc":
        return float(balanced_accuracy_score(y, pred))
    if args.selection_metric == "macro_f1":
        return float(precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[2])
    raise ValueError(f"unknown selection metric {args.selection_metric}")


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


def class_weights(
    y: np.ndarray,
    n_classes: int,
    mode: str,
    power: float,
    effective_beta: float,
    weight_min: float,
    weight_max: float,
) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    if mode == "power":
        weights = (len(y) / (n_classes * counts)) ** power
    elif mode == "effective":
        effective_num = 1.0 - np.power(effective_beta, counts)
        weights = (1.0 - effective_beta) / np.maximum(effective_num, 1e-8)
        weights = np.power(weights, power)
    else:
        raise ValueError(f"unknown class weight mode {mode}")
    weights = weights / weights.mean()
    if weight_min > 0 or weight_max > 0:
        lower = weight_min if weight_min > 0 else float(weights.min())
        upper = weight_max if weight_max > 0 else float(weights.max())
        weights = np.clip(weights, lower, upper)
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


def print_head_diagnostics(
    labels: list[dict[str, str] | None],
    pred_rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
) -> None:
    indices = [index for index, label in enumerate(labels) if label is not None]
    y_true = [labels[index][HEAD] for index in indices]  # type: ignore[index]
    y_base = [pred_rows[index][HEAD] for index in indices]
    y_model = [str(predictions[HEAD][index]) for index in indices]
    print("diagnostic,head,source,accuracy,balanced_acc,macro_f1,weighted_f1")
    print_diagnostic_row("baseline", y_true, y_base)
    print_diagnostic_row("raw_model", y_true, y_model)
    print_per_class_rows("baseline", y_true, y_base)
    print_per_class_rows("raw_model", y_true, y_model)
    print_confusion_rows("baseline", y_true, y_base)
    print_confusion_rows("raw_model", y_true, y_model)


def print_diagnostic_row(source: str, y_true: list[str], y_pred: list[str]) -> None:
    print(
        f"diagnostic,{HEAD},{source},{accuracy_score(y_true, y_pred):.6f},"
        f"{balanced_accuracy_score(y_true, y_pred):.6f},"
        f"{macro_f1(y_true, y_pred):.6f},{weighted_f1(y_true, y_pred):.6f}",
        flush=True,
    )


def print_per_class_rows(source: str, y_true: list[str], y_pred: list[str]) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    for label, p_value, r_value, f1_value, support_value in zip(labels, precision, recall, f1, support):
        print(
            f"diagnostic_class,{HEAD},{source},{label},precision={p_value:.6f},"
            f"recall={r_value:.6f},f1={f1_value:.6f},support={int(support_value)}",
            flush=True,
        )


def print_confusion_rows(source: str, y_true: list[str], y_pred: list[str]) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"confusion_labels,{HEAD},{source}," + ",".join(labels), flush=True)
    for label, row in zip(labels, matrix):
        print(f"confusion,{HEAD},{source},true={label}," + ",".join(str(int(value)) for value in row), flush=True)


def print_best_rows(score_by_margin: dict[float, dict[str, Any]], baseline: dict[str, Any]) -> None:
    if not score_by_margin:
        return
    best_macro = max(score_by_margin, key=lambda margin: score_by_margin[margin]["macro_score"])
    best_eff = max(score_by_margin, key=lambda margin: component_summary(score_by_margin[margin])["effectiveness"])
    for label, margin in [("best_by_macro", best_macro), ("best_by_effectiveness", best_eff)]:
        score = score_by_margin[margin]
        summary = component_summary(score)
        print(
            f"{label},margin={margin:g},macro={score['macro_score']:.6f},"
            f"delta={score['macro_score'] - baseline['macro_score']:+.6f},"
            f"effectiveness={summary['effectiveness']:.6f},wins={video_wins(score, baseline)}",
            flush=True,
        )


def print_video_deltas(margin: float, score: dict[str, Any], baseline: dict[str, Any]) -> None:
    print(f"video_delta_for_margin={margin:g}")
    print("video_key,macro_delta,effectiveness_delta,new_macro,base_macro,new_effectiveness,base_effectiveness")
    for key in sorted(score["by_video"]):
        item = score["by_video"][key]
        base = baseline["by_video"][key]
        print(
            f"{key},{item['final_score'] - base['final_score']:+.6f},"
            f"{item['score_effectiveness'] - base['score_effectiveness']:+.6f},"
            f"{item['final_score']:.6f},{base['final_score']:.6f},"
            f"{item['score_effectiveness']:.6f},{base['score_effectiveness']:.6f}",
            flush=True,
        )


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2])


def weighted_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2])


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
