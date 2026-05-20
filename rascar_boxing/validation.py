"""Dataset and submission validation helpers."""

from __future__ import annotations

from pathlib import Path

from .constants import ALLOWED_VALUES, SUBMISSION_COLUMNS
from .io import as_int, read_csv_header, read_csv_rows


def validate_video_files(
    data_root: Path,
    video_csv_paths: list[Path],
    check_video_props: bool = False,
) -> list[str]:
    errors: list[str] = []
    for csv_path in video_csv_paths:
        rows = read_csv_rows(csv_path)
        for row_index, row in enumerate(rows, start=2):
            rel_path = row.get("video_path", "")
            full_path = data_root / rel_path
            if not full_path.exists():
                errors.append(f"{csv_path}:{row_index}: missing video {full_path}")
                continue
            if not full_path.is_file():
                errors.append(f"{csv_path}:{row_index}: video path is not a file {full_path}")
                continue
            if check_video_props:
                errors.extend(_validate_video_props(full_path, row, csv_path, row_index))
    return errors


def _validate_video_props(
    video_path: Path,
    row: dict[str, str],
    csv_path: Path,
    row_index: int,
) -> list[str]:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001 - optional dependency boundary.
        return [f"OpenCV is required for --check-video-props: {exc}"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [f"{csv_path}:{row_index}: OpenCV cannot open {video_path}"]

    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    errors: list[str] = []
    expected_frame_count = as_int(row["frame_count"], "frame_count")
    expected_width = as_int(row["width"], "width")
    expected_height = as_int(row["height"], "height")

    if frame_count != expected_frame_count:
        errors.append(
            f"{csv_path}:{row_index}: frame_count mismatch for {video_path}: "
            f"csv={expected_frame_count}, video={frame_count}"
        )
    if width != expected_width or height != expected_height:
        errors.append(
            f"{csv_path}:{row_index}: size mismatch for {video_path}: "
            f"csv={expected_width}x{expected_height}, video={width}x{height}"
        )
    if fps <= 0:
        errors.append(f"{csv_path}:{row_index}: invalid FPS for {video_path}: {fps}")
    return errors


def validate_submission(
    submission_path: Path,
    sample_path: Path,
    strict_id_metadata: bool = True,
) -> list[str]:
    errors: list[str] = []
    header = read_csv_header(submission_path)
    if header != SUBMISSION_COLUMNS:
        errors.append(f"submission columns mismatch: got {header}, expected {SUBMISSION_COLUMNS}")
        return errors

    sample_rows = read_csv_rows(sample_path)
    rows = read_csv_rows(submission_path)
    if len(rows) != len(sample_rows):
        errors.append(f"row count mismatch: got {len(rows)}, expected {len(sample_rows)}")

    sample_ids = {row["id"] for row in sample_rows}
    ids = [row["id"] for row in rows]
    duplicate_ids = sorted({id_ for id_ in ids if ids.count(id_) > 1}, key=int)
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids[:20]}")

    id_set = set(ids)
    if id_set != sample_ids:
        missing = sorted(sample_ids - id_set, key=int)
        extra = sorted(id_set - sample_ids, key=int)
        if missing:
            errors.append(f"missing ids: {missing[:20]}")
        if extra:
            errors.append(f"unexpected ids: {extra[:20]}")

    if strict_id_metadata:
        sample_by_id = {row["id"]: row for row in sample_rows}
        for row in rows:
            ref = sample_by_id.get(row["id"])
            if ref is None:
                continue
            for col in ["video_id", "agn_index", "video_key"]:
                if row[col] != ref[col]:
                    errors.append(
                        f"id={row['id']}: {col} mismatch: got {row[col]!r}, expected {ref[col]!r}"
                    )

    for row_number, row in enumerate(rows, start=2):
        try:
            frame = as_int(row["frame"], "frame")
            if frame < 0:
                errors.append(f"{submission_path}:{row_number}: frame must be >= 0")
        except ValueError as exc:
            errors.append(f"{submission_path}:{row_number}: {exc}")

        for col, allowed in ALLOWED_VALUES.items():
            value = row[col]
            if value not in allowed:
                errors.append(
                    f"{submission_path}:{row_number}: invalid {col}={value!r}; "
                    f"allowed={sorted(allowed)}"
                )
    return errors


def validate_dataset(data_root: Path, check_video_props: bool = False) -> list[str]:
    errors: list[str] = []
    required_files = [
        data_root / "train/videos.csv",
        data_root / "train/punches.csv",
        data_root / "test/videos.csv",
        data_root / "sample_submission.csv",
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing required file: {path}")
    if errors:
        return errors

    errors.extend(
        validate_video_files(
            data_root,
            [data_root / "train/videos.csv", data_root / "test/videos.csv"],
            check_video_props=check_video_props,
        )
    )
    return errors

