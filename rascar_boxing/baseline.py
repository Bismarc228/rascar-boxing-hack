"""Simple submission baselines."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .constants import SUBMISSION_COLUMNS
from .io import normalize_bool, read_csv_rows, write_csv_rows


ATTR_COLUMNS = ["fighter", "punch_type", "hand", "target", "effectiveness"]


def make_submission(
    data_root: Path,
    output_path: Path,
    strategy: str = "sample",
    clear_mode: str = "strategy",
) -> None:
    sample_rows = read_csv_rows(data_root / "sample_submission.csv")
    train_rows = read_csv_rows(data_root / "train/punches.csv")

    if strategy == "sample":
        rows = [{col: row[col] for col in SUBMISSION_COLUMNS} for row in sample_rows]
    elif strategy == "all_false":
        rows = [{col: row[col] for col in SUBMISSION_COLUMNS} for row in sample_rows]
        for row in rows:
            row["clear"] = "false"
    elif strategy == "train_priors":
        priors = most_common_attributes(train_rows, clear_only=True)
        rows = []
        for sample_row in sample_rows:
            row = {col: sample_row[col] for col in SUBMISSION_COLUMNS}
            for col, value in priors.items():
                row[col] = value
            rows.append(row)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    if clear_mode != "strategy":
        if clear_mode == "sample":
            sample_by_id = {row["id"]: row for row in sample_rows}
            for row in rows:
                row["clear"] = normalize_bool(sample_by_id[row["id"]]["clear"])
        elif clear_mode == "all_true":
            for row in rows:
                row["clear"] = "true"
        elif clear_mode == "all_false":
            for row in rows:
                row["clear"] = "false"
        else:
            raise ValueError(f"Unknown clear_mode {clear_mode!r}")

    write_csv_rows(output_path, rows, SUBMISSION_COLUMNS)


def most_common_attributes(rows: list[dict[str, str]], clear_only: bool = True) -> dict[str, str]:
    filtered = [row for row in rows if row.get("clear") == "true"] if clear_only else rows
    if not filtered:
        filtered = rows

    values: dict[str, str] = {}
    for col in ATTR_COLUMNS:
        counter = Counter(row[col] for row in filtered)
        values[col] = counter.most_common(1)[0][0]
    return values

