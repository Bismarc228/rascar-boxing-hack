#!/usr/bin/env python3
"""Build a CUDA-decoded contact sheet for submission clear-drop diffs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from rascar_boxing.io import as_int, read_csv_rows, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="clear_drops")
    parser.add_argument("--frame-offsets", default="-4,0,4")
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--jpeg-quality", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = [int(item.strip()) for item in args.frame_offsets.split(",") if item.strip()]
    if 0 not in offsets:
        offsets.append(0)
    offsets = sorted(set(offsets))

    videos = load_video_metadata(args.data_root)
    diagnostics = load_diagnostics(args.diagnostics) if args.diagnostics else {}
    changed = clear_drop_rows(args.base, args.candidate)
    if not changed:
        raise SystemExit("no clear=true -> clear=false rows found")

    audit_rows = []
    tile_rows = []
    for row_index, row in enumerate(changed, start=1):
        video = videos[row["video_key"]]
        frame = as_int(row["frame"], "frame")
        frame_count = as_int(video["frame_count"], "frame_count")
        row_tiles = []
        for offset in offsets:
            wanted_frame = min(max(0, frame + offset), max(0, frame_count - 1))
            image_path = args.output_dir / (
                f"{args.prefix}_{row_index:03d}_id{row['id']}_"
                f"{row['video_key']}_f{wanted_frame}.jpg"
            )
            extract_frame_cuda(
                args.data_root / video["video_path"],
                wanted_frame,
                image_path,
                args.jpeg_quality,
            )
            row_tiles.append((offset, wanted_frame, image_path))
        diag = diagnostics.get(row["id"], {})
        audit_rows.append(
            {
                "id": row["id"],
                "video_key": row["video_key"],
                "frame": row["frame"],
                "fighter": row["fighter"],
                "punch_type": row["punch_type"],
                "hand": row["hand"],
                "target": row["target"],
                "effectiveness": row["effectiveness"],
                "p_keep": diag.get("p_keep", ""),
                "frames": ";".join(str(frame_number) for _offset, frame_number, _path in row_tiles),
            }
        )
        tile_rows.append((row, diag, row_tiles))

    rows_csv = args.output_dir / f"{args.prefix}_rows.csv"
    write_csv_rows(rows_csv, audit_rows, audit_columns())
    contact_sheet = args.output_dir / f"{args.prefix}_contact_sheet.jpg"
    build_contact_sheet(tile_rows, offsets, contact_sheet, args.tile_width)
    print(f"changed_rows={len(changed)} rows_csv={rows_csv} contact_sheet={contact_sheet}")
    return 0


def load_video_metadata(data_root: Path) -> dict[str, dict[str, str]]:
    rows = []
    for split in ["train", "test"]:
        path = data_root / split / "videos.csv"
        if path.exists():
            rows.extend(read_csv_rows(path))
    return {row["video_key"]: row for row in rows}


def load_diagnostics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def clear_drop_rows(base_path: Path, candidate_path: Path) -> list[dict[str, str]]:
    base_rows = {row["id"]: row for row in read_csv_rows(base_path)}
    candidate_rows = {row["id"]: row for row in read_csv_rows(candidate_path)}
    output = []
    for row_id, base in base_rows.items():
        candidate = candidate_rows.get(row_id)
        if candidate is None:
            continue
        if base.get("clear") == "true" and candidate.get("clear") == "false":
            output.append(base)
    output.sort(key=lambda row: as_int(row["id"], "id"))
    return output


def extract_frame_cuda(video_path: Path, frame: int, output_path: Path, quality: int) -> None:
    select_expr = f"eq(n\\,{frame})"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-hwaccel",
        "cuda",
        "-c:v",
        "hevc_cuvid",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-vf",
        f"select={select_expr}",
        "-frames:v",
        "1",
        "-q:v",
        str(quality),
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def build_contact_sheet(
    tile_rows: list[tuple[dict[str, str], dict[str, str], list[tuple[int, int, Path]]]],
    offsets: list[int],
    output_path: Path,
    tile_width: int,
) -> None:
    font = ImageFont.load_default()
    header_h = 42
    gap = 8
    prepared_rows = []
    tile_h = None
    for row, diag, tiles in tile_rows:
        prepared = []
        for offset, frame, path in tiles:
            image = Image.open(path).convert("RGB")
            ratio = tile_width / float(image.width)
            resized = image.resize((tile_width, int(round(image.height * ratio))), Image.Resampling.LANCZOS)
            tile_h = resized.height
            annotated = Image.new("RGB", (tile_width, header_h + resized.height), "black")
            annotated.paste(resized, (0, header_h))
            draw = ImageDraw.Draw(annotated)
            title = (
                f"id={row['id']} {row['video_key']} f={frame} off={offset:+d} "
                f"{row['effectiveness']} p={diag.get('p_keep', '')}"
            )
            draw.text((6, 5), title[:92], fill="white", font=font)
            subtitle = f"{row['fighter']} {row['punch_type']} {row['hand']} -> {row['target']}"
            draw.text((6, 22), subtitle[:92], fill="white", font=font)
            prepared.append(annotated)
        prepared_rows.append(prepared)

    if tile_h is None:
        raise SystemExit("no images to compose")
    width = len(offsets) * tile_width + (len(offsets) - 1) * gap
    row_h = header_h + tile_h
    height = len(prepared_rows) * row_h + (len(prepared_rows) - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for prepared in prepared_rows:
        x = 0
        for image in prepared:
            sheet.paste(image, (x, y))
            x += tile_width + gap
        y += row_h + gap
    sheet.save(output_path, quality=92)


def audit_columns() -> list[str]:
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
        "frames",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
