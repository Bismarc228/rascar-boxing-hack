"""Deterministic fight-level train/validation split helpers."""

from __future__ import annotations

import random
from pathlib import Path

from .io import read_csv_rows


def make_fight_split(
    videos_csv: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rows = read_csv_rows(videos_csv)
    groups: dict[str, list[str]] = {}
    for row in rows:
        group_key = "|".join(
            [
                row["dataset_type"],
                row["data_root"],
                row["fight_index"],
                row["fight_folder"],
            ]
        )
        groups.setdefault(group_key, []).append(row["video_key"])

    group_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_val_groups = max(1, round(len(group_keys) * val_fraction))
    val_groups = set(group_keys[:n_val_groups])

    train_keys: list[str] = []
    val_keys: list[str] = []
    for group_key in sorted(groups):
        target = val_keys if group_key in val_groups else train_keys
        target.extend(sorted(groups[group_key]))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_video_keys.txt").write_text("\n".join(train_keys) + "\n", encoding="utf-8")
    (output_dir / "val_video_keys.txt").write_text("\n".join(val_keys) + "\n", encoding="utf-8")
    (output_dir / "val_groups.txt").write_text("\n".join(sorted(val_groups)) + "\n", encoding="utf-8")
    return train_keys, val_keys

