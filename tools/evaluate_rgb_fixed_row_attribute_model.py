#!/usr/bin/env python3
"""Evaluate cached RGB clip features for fixed-row attribute replacement."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rascar_boxing.constants import SUBMISSION_COLUMNS
from rascar_boxing.io import as_int, group_by, read_csv_rows, write_csv_rows
from rascar_boxing.metric import match_events, score_predictions
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, PunchCandidate, score_pose_tracks, select_candidates
from tools.evaluate_pose_selection_variants import video_wins


ATTR_COLUMNS = ["punch_type", "effectiveness", "hand", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--rgb-feature-cache", type=Path, required=True)
    parser.add_argument("--pool-min-score", type=float, default=0.0)
    parser.add_argument("--pool-nms-frames", type=int, default=2)
    parser.add_argument("--max-candidates-per-video", type=int, default=1800)
    parser.add_argument("--candidate-match-window", type=int, default=6)
    parser.add_argument("--label-window", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--logreg-c", type=float, default=0.35)
    parser.add_argument("--head-type", choices=["logreg", "torch_linear", "torch_mlp"], default="logreg")
    parser.add_argument(
        "--train-columns",
        default=",".join(ATTR_COLUMNS),
        help="Comma-separated attribute heads to train; untrained heads keep input values.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-hidden-dim", type=int, default=192)
    parser.add_argument("--torch-depth", type=int, default=2)
    parser.add_argument("--torch-dropout", type=float, default=0.15)
    parser.add_argument("--torch-epochs", type=int, default=120)
    parser.add_argument("--torch-batch-size", type=int, default=256)
    parser.add_argument("--torch-lr", type=float, default=3e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-3)
    parser.add_argument("--torch-seed", type=int, default=17)
    parser.add_argument("--torch-class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--torch-pca-components", type=int, default=0)
    parser.add_argument("--effectiveness-margins", default="0.0,0.1,0.2,0.3")
    parser.add_argument(
        "--margin-columns",
        default="effectiveness",
        help="Comma-separated attribute columns for confidence-margin variants.",
    )
    parser.add_argument("--write-oof-rows", type=Path)
    parser.add_argument("--write-effectiveness-margin", type=float)
    parser.add_argument("--write-margin-column", choices=ATTR_COLUMNS, default="effectiveness")
    parser.add_argument(
        "--write-variant",
        choices=["punch_type", "effectiveness", "ptype_eff", "hand_target", "all_attrs"],
        default="all_attrs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.predictions) if row.get("clear") == "true"]
    keys = sorted({row["video_key"] for row in pred_rows})
    videos = read_csv_rows(args.data_root / "train/videos.csv")
    video_by_key = {row["video_key"]: row for row in videos}
    missing = sorted(set(keys) - set(video_by_key))
    if missing:
        raise RuntimeError(f"Prediction keys are not train videos: {missing}")
    gt_rows = [
        row
        for row in read_csv_rows(args.data_root / "train/punches.csv")
        if row.get("clear") == "true" and row["video_key"] in set(keys)
    ]
    baseline = score_predictions(gt_rows, pred_rows)
    print_score("baseline", baseline, len(pred_rows), 0)

    features = load_rgb_features(args.rgb_feature_cache)
    candidates_by_key, feature_by_key = reconstruct_feature_index(args, keys, features)
    x, matched = build_features(args, pred_rows, video_by_key, candidates_by_key, feature_by_key)
    labels = matched_attribute_labels(pred_rows, gt_rows, args.label_window)
    groups = np.asarray([fight_group(video_by_key[row["video_key"]]) for row in pred_rows])
    print(
        f"rgb_feature_rows={len(features)} pred_rows={len(pred_rows)} "
        f"matched={int(matched.sum())} match_rate={float(matched.mean()):.4f} "
        f"x_shape={x.shape}",
        flush=True,
    )

    train_columns = [column.strip() for column in args.train_columns.split(",") if column.strip()]
    for column in train_columns:
        if column not in ATTR_COLUMNS:
            raise ValueError(f"unknown train column {column}")
    if args.head_type.startswith("torch_"):
        print(f"torch_head_device={resolve_device(args.device)} train_columns={','.join(train_columns)}", flush=True)
    predictions, probabilities = oof_attribute_predictions(args, x, labels, groups, pred_rows, matched, train_columns)
    variants = {
        "punch_type": ["punch_type"],
        "effectiveness": ["effectiveness"],
        "ptype_eff": ["punch_type", "effectiveness"],
        "hand_target": ["hand", "target"],
        "all_attrs": ATTR_COLUMNS,
    }
    print("variant,score,delta,type,effectiveness,hand,target,wins,n_changed")
    rows_by_variant = {}
    for name, columns in variants.items():
        rows, changed = apply_predictions(pred_rows, predictions, columns)
        rows_by_variant[name] = rows
        score = score_predictions(gt_rows, rows)
        summary = attr_summary(score)
        print(
            f"{name},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:.6f},"
            f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
            f"{summary['hand']:.6f},{summary['target']:.6f},"
            f"{video_wins(score, baseline)},{changed}",
            flush=True,
        )
    margin_columns = [column.strip() for column in args.margin_columns.split(",") if column.strip()]
    for column in margin_columns:
        if column not in ATTR_COLUMNS:
            raise ValueError(f"unknown margin column {column}")
        for margin in parse_floats(args.effectiveness_margins):
            name = f"{column}_margin_{margin:g}"
            rows, changed = apply_attribute_margin(pred_rows, predictions, probabilities, column, margin)
            rows_by_variant[name] = rows
            score = score_predictions(gt_rows, rows)
            summary = attr_summary(score)
            print(
                f"{name},{score['macro_score']:.6f},{score['macro_score'] - baseline['macro_score']:.6f},"
                f"{summary['punch_type']:.6f},{summary['effectiveness']:.6f},"
                f"{summary['hand']:.6f},{summary['target']:.6f},"
                f"{video_wins(score, baseline)},{changed}",
                flush=True,
            )
    if args.write_oof_rows:
        variant_name = args.write_variant
        if args.write_effectiveness_margin is not None:
            variant_name = f"{args.write_margin_column}_margin_{args.write_effectiveness_margin:g}"
        if variant_name not in rows_by_variant:
            raise KeyError(f"variant {variant_name} was not evaluated")
        write_csv_rows(args.write_oof_rows, rows_by_variant[variant_name], SUBMISSION_COLUMNS)
        print(
            f"wrote_oof_rows={args.write_oof_rows} "
            f"variant={variant_name} n_rows={len(rows_by_variant[variant_name])}",
            flush=True,
        )
    print_label_counts(labels)
    return 0


def load_rgb_features(path: Path) -> np.ndarray:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    if features.ndim != 3:
        raise ValueError(f"expected feature tensor [N,T,D], got {features.shape}")
    return features


def reconstruct_feature_index(
    args: argparse.Namespace,
    keys: list[str],
    features: np.ndarray,
) -> tuple[dict[str, list[PunchCandidate]], dict[str, np.ndarray]]:
    candidates_by_key: dict[str, list[PunchCandidate]] = {}
    feature_by_key: dict[str, np.ndarray] = {}
    offset = 0
    for key in keys:
        raw = score_pose_tracks(args.tracks_dir / f"{key}.jsonl", PoseHeuristicConfig(min_score=0.0))
        pool = select_candidates(
            raw,
            PoseHeuristicConfig(
                min_score=args.pool_min_score,
                nms_frames=args.pool_nms_frames,
                nms_group_mode="fighter_hand",
                cross_nms_frames=0,
            ),
            args.max_candidates_per_video,
        )
        n_items = len(pool)
        candidates_by_key[key] = pool
        feature_by_key[key] = features[offset : offset + n_items]
        offset += n_items
    if offset != len(features):
        raise ValueError(f"reconstructed {offset} candidates but cache has {len(features)} features")
    return candidates_by_key, feature_by_key


def build_features(
    args: argparse.Namespace,
    pred_rows: list[dict[str, str]],
    video_by_key: dict[str, dict[str, str]],
    candidates_by_key: dict[str, list[PunchCandidate]],
    feature_by_key: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    matched = []
    dim = next(iter(feature_by_key.values())).shape[-1]
    zero_rgb = np.zeros(dim * 2, dtype=np.float32)
    for row in pred_rows:
        key = row["video_key"]
        index = nearest_candidate_index(row, candidates_by_key[key], args.candidate_match_window)
        if index is None:
            rgb = zero_rgb
            matched.append(False)
        else:
            clip = feature_by_key[key][index]
            rgb = np.concatenate([clip.mean(axis=0), clip.std(axis=0)]).astype(np.float32)
            matched.append(True)
        rows.append(np.concatenate([rgb, row_context_features(row, video_by_key[key])]).astype(np.float32))
    return np.stack(rows), np.asarray(matched, dtype=bool)


def nearest_candidate_index(
    row: dict[str, str],
    candidates: list[PunchCandidate],
    window: int,
) -> int | None:
    frame = as_int(row["frame"], "frame")
    passes = [
        lambda item: item.fighter == row["fighter"] and item.hand == row["hand"],
        lambda item: item.fighter == row["fighter"],
        lambda item: True,
    ]
    for predicate in passes:
        best_index = None
        best_dist = window + 1
        for index, candidate in enumerate(candidates):
            if not predicate(candidate):
                continue
            dist = abs(candidate.frame - frame)
            if dist <= window and dist < best_dist:
                best_index = index
                best_dist = dist
        if best_index is not None:
            return best_index
    return None


def row_context_features(row: dict[str, str], video: dict[str, str]) -> np.ndarray:
    values = [
        1.0 if row["fighter"] == "red" else 0.0,
        1.0 if row["hand"] == "left" else 0.0,
        1.0 if row["target"] == "head" else 0.0,
        1.0 if row["punch_type"] == "jab" else 0.0,
        1.0 if row["punch_type"] == "cross" else 0.0,
        1.0 if row["punch_type"] == "hook" else 0.0,
        1.0 if row["punch_type"] == "uppercut" else 0.0,
        1.0 if row["effectiveness"] == "landed" else 0.0,
        1.0 if row["effectiveness"] == "blocked" else 0.0,
        1.0 if row["effectiveness"] in {"miss", "missed"} else 0.0,
        as_int(row["frame"], "frame") / max(1.0, float(video["frame_count"])),
    ]
    return np.asarray(values, dtype=np.float32)


def matched_attribute_labels(
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    label_window: int,
) -> list[dict[str, str] | None]:
    labels: list[dict[str, str] | None] = [None] * len(pred_rows)
    pred_by_key = group_by(pred_rows, "video_key")
    gt_by_key = group_by(gt_rows, "video_key")
    global_indices_by_key: dict[str, list[int]] = {}
    for index, row in enumerate(pred_rows):
        global_indices_by_key.setdefault(row["video_key"], []).append(index)
    for key, rows in pred_by_key.items():
        for match in match_events(gt_by_key.get(key, []), rows):
            pred = rows[match.pred_index]
            gt = gt_by_key[key][match.gt_index]
            if abs(as_int(pred["frame"], "frame") - as_int(gt["frame"], "frame")) > label_window:
                continue
            labels[global_indices_by_key[key][match.pred_index]] = gt
    return labels


def oof_attribute_predictions(
    args: argparse.Namespace,
    x: np.ndarray,
    labels: list[dict[str, str] | None],
    groups: np.ndarray,
    pred_rows: list[dict[str, str]],
    matched: np.ndarray,
    train_columns: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, float]]]]:
    output = {column: np.asarray([row[column] for row in pred_rows], dtype=object) for column in ATTR_COLUMNS}
    probabilities = {
        column: [{row[column]: 1.0} for row in pred_rows]
        for column in ATTR_COLUMNS
    }
    trainable = np.asarray([label is not None for label in labels], dtype=bool) & matched
    for group in sorted(set(groups.tolist())):
        valid = groups == group
        train = (groups != group) & trainable
        valid_predict = valid & matched
        if train.sum() < 20 or valid_predict.sum() == 0:
            print(f"fold={group} train={int(train.sum())} valid={int(valid_predict.sum())} skipped", flush=True)
            continue
        for column in train_columns:
            values = [labels[index][column] for index in np.where(train)[0]]  # type: ignore[index]
            encoder = LabelEncoder()
            y = encoder.fit_transform(values)
            if len(encoder.classes_) == 1:
                output[column][valid_predict] = encoder.classes_[0]
                for index in np.where(valid_predict)[0]:
                    probabilities[column][index] = {str(encoder.classes_[0]): 1.0}
                continue
            if args.head_type.startswith("torch_"):
                pred_encoded, proba = fit_torch_mlp_classifier(
                    args,
                    x[train],
                    y,
                    x[valid_predict],
                    len(encoder.classes_),
                    stable_fold_seed(group, column, args.torch_seed),
                )
            else:
                n_components = min(args.pca_components, train.sum() - 1, x.shape[1])
                model = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=n_components, random_state=42),
                    LogisticRegression(
                        C=args.logreg_c,
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=42,
                    ),
                )
                model.fit(x[train], y)
                pred_encoded = model.predict(x[valid_predict])
                proba = model.predict_proba(x[valid_predict])
            pred_values = encoder.inverse_transform(pred_encoded)
            output[column][valid_predict] = pred_values
            class_values = [str(value) for value in encoder.classes_]
            for index, row_probs in zip(np.where(valid_predict)[0], proba):
                probabilities[column][index] = {
                    class_value: float(prob)
                    for class_value, prob in zip(class_values, row_probs)
                }
        print(f"fold={group} train={int(train.sum())} valid={int(valid_predict.sum())}", flush=True)
    return output, probabilities


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout) for _ in range(depth)])
        self.output = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


def fit_torch_mlp_classifier(
    args: argparse.Namespace,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_torch_seed(seed)
    device = resolve_device(args.device)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    if args.torch_pca_components > 0:
        train_x, valid_x = project_torch_pca(args, train_x, valid_x, device)
    if args.head_type == "torch_linear":
        model: nn.Module = nn.Linear(train_x.shape[1], n_classes)
    else:
        model = ResidualMLP(
            input_dim=train_x.shape[1],
            hidden_dim=args.torch_hidden_dim,
            n_classes=n_classes,
            depth=args.torch_depth,
            dropout=args.torch_dropout,
        )
    model = model.to(device)
    class_counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.from_numpy(class_weights).to(device) if args.torch_class_weight == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.torch_lr, weight_decay=args.torch_weight_decay)
    train_tensor = torch.from_numpy(train_x.astype(np.float32))
    target_tensor = torch.from_numpy(train_y.astype(np.int64))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor, target_tensor),
        batch_size=args.torch_batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    model.train()
    for _epoch in range(args.torch_epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(valid_x.astype(np.float32)).to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba.argmax(axis=1).astype(np.int64), proba.astype(np.float32)


def project_torch_pca(
    args: argparse.Namespace,
    train_x: np.ndarray,
    valid_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(args.torch_pca_components, train_x.shape[0] - 1, train_x.shape[1])
    train_tensor = torch.from_numpy(train_x.astype(np.float32)).to(device)
    valid_tensor = torch.from_numpy(valid_x.astype(np.float32)).to(device)
    with torch.inference_mode():
        _u, _s, v = torch.pca_lowrank(train_tensor, q=n_components, center=False, niter=5)
        train_projected = (train_tensor @ v[:, :n_components]).cpu().numpy()
        valid_projected = (valid_tensor @ v[:, :n_components]).cpu().numpy()
    return train_projected.astype(np.float32), valid_projected.astype(np.float32)


def apply_predictions(
    rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    columns: list[str],
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        for column in columns:
            item[column] = str(predictions[column][index])
        if any(item[column] != row[column] for column in columns):
            changed += 1
        output.append(item)
    return output, changed


def apply_attribute_margin(
    rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, list[dict[str, float]]],
    column: str,
    margin: float,
) -> tuple[list[dict[str, str]], int]:
    output = []
    changed = 0
    for index, row in enumerate(rows):
        item = {col: row.get(col, "") for col in SUBMISSION_COLUMNS}
        pred_value = str(predictions[column][index])
        current_value = row[column]
        probs = probabilities[column][index]
        pred_prob = float(probs.get(pred_value, 0.0))
        current_prob = float(probs.get(current_value, 0.0))
        if pred_value != current_value and pred_prob - current_prob >= margin:
            item[column] = pred_value
            changed += 1
        output.append(item)
    return output, changed


def print_score(label: str, score: dict[str, Any], n_rows: int, wins: int) -> None:
    summary = attr_summary(score)
    print(
        f"{label}: score={score['macro_score']:.6f},"
        f"type={summary['punch_type']:.6f},eff={summary['effectiveness']:.6f},"
        f"hand={summary['hand']:.6f},target={summary['target']:.6f},wins={wins},n={n_rows}",
        flush=True,
    )


def attr_summary(score: dict[str, Any]) -> dict[str, float]:
    by_video = score["by_video"]
    return {
        "punch_type": float(np.mean([item["score_punch_type"] for item in by_video.values()])),
        "effectiveness": float(np.mean([item["score_effectiveness"] for item in by_video.values()])),
        "hand": float(np.mean([item["score_hand"] for item in by_video.values()])),
        "target": float(np.mean([item["score_target"] for item in by_video.values()])),
    }


def print_label_counts(labels: list[dict[str, str] | None]) -> None:
    usable = [label for label in labels if label is not None]
    print(f"labeled_rows={len(usable)}")
    for column in ATTR_COLUMNS:
        counts: dict[str, int] = {}
        for label in usable:
            counts[label[column]] = counts.get(label[column], 0) + 1
        print("labels_" + column + "=" + ",".join(f"{key}:{value}" for key, value in sorted(counts.items())))


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def fight_group(video: dict[str, str]) -> str:
    return "|".join([video["dataset_type"], video["data_root"], video["fight_index"], video["fight_folder"]])


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        if torch.cuda.is_available() and visible == "1":
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(value)


def stable_fold_seed(group: str, column: str, seed: int) -> int:
    value = seed
    for char in f"{group}|{column}":
        value = (value * 131 + ord(char)) % 2_147_483_647
    return value


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
