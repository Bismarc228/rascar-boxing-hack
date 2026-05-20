#!/usr/bin/env python3
"""Download the boxing videos referenced by the train/test CSV files.

The Kaggle archive contains only metadata. The actual videos are in three
public Yandex Disk folders. This script resolves direct download URLs through
the Yandex public resources API and stores files under data/raw/<video_path>.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


API_BASE = "https://cloud-api.yandex.net/v1/disk/public/resources"

SOURCES = {
    "Турнир Бокс": "https://disk.yandex.ru/d/2unA0OZ2yA0ClA",
    "Турнир Бокс 2": "https://disk.yandex.ru/d/rOAhWV1x2ezWmg",
    "бокс": "https://disk.yandex.ru/d/txJP8PEGTMHTIQ",
}


@dataclass(frozen=True)
class VideoTask:
    video_path: str
    public_url: str
    remote_path: str
    dest: Path


class Progress:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total_bytes = 0
        self.done_bytes = 0
        self.active: dict[str, int] = {}
        self.completed = 0
        self.skipped = 0
        self.failed = 0
        self.stop = False

    def monitor(self, total_files: int, interval: float = 30.0) -> None:
        start = time.monotonic()
        while True:
            time.sleep(interval)
            with self.lock:
                if self.stop:
                    return
                active_bytes = sum(self.active.values())
                current = self.done_bytes + active_bytes
                total = self.total_bytes
                elapsed = max(time.monotonic() - start, 1e-6)
                rate = current / elapsed
                pct = (current / total * 100.0) if total else 0.0
                print(
                    f"[progress] files={self.completed + self.skipped}/{total_files} "
                    f"active={len(self.active)} failed={self.failed} "
                    f"bytes={fmt_bytes(current)}/{fmt_bytes(total)} "
                    f"({pct:.1f}%) rate={fmt_bytes(rate)}/s",
                    flush=True,
                )


def fmt_bytes(value: float | int) -> str:
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def api_json(endpoint: str, params: dict[str, str | int], retries: int = 8) -> dict:
    url = f"{API_BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rascar-box-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(15 * attempt, 90)
            elif 500 <= exc.code < 600:
                delay = min(2**attempt, 30)
            else:
                break
            if attempt == retries:
                break
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"API request failed for {url}: {last_error}")


def collect_tasks(csv_paths: list[Path], output_root: Path) -> list[VideoTask]:
    seen: set[str] = set()
    tasks: list[VideoTask] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                video_path = row["video_path"]
                if video_path in seen:
                    continue
                seen.add(video_path)

                root, _, rest = video_path.partition("/")
                if not rest or root not in SOURCES:
                    raise ValueError(f"Cannot map video_path to Yandex source: {video_path}")

                tasks.append(
                    VideoTask(
                        video_path=video_path,
                        public_url=SOURCES[root],
                        remote_path="/" + rest,
                        dest=output_root / video_path,
                    )
                )
    return sorted(tasks, key=lambda task: task.video_path)


def get_metadata(task: VideoTask) -> dict:
    return api_json("", {"public_key": task.public_url, "path": task.remote_path})


def resolve_sizes(tasks: list[VideoTask]) -> dict[str, int | None]:
    by_parent: dict[tuple[str, str], list[VideoTask]] = {}
    for task in tasks:
        parent = str(Path(task.remote_path).parent)
        if parent == ".":
            parent = "/"
        by_parent.setdefault((task.public_url, parent), []).append(task)

    sizes: dict[str, int | None] = {}
    for index, ((public_url, parent), grouped_tasks) in enumerate(sorted(by_parent.items()), 1):
        data = api_json(
            "",
            {
                "public_key": public_url,
                "path": parent,
                "limit": 1000,
            },
        )
        items = data.get("_embedded", {}).get("items", [])
        items_by_path = {item.get("path"): item for item in items}
        print(
            f"[meta {index:02d}/{len(by_parent):02d}] {parent} "
            f"({len(grouped_tasks)} needed files)",
            flush=True,
        )
        for task in grouped_tasks:
            item = items_by_path.get(task.remote_path)
            if item is None:
                item = get_metadata(task)
            size = item.get("size")
            sizes[task.video_path] = int(size) if size is not None else None
        time.sleep(0.5)
    return sizes


def get_download_href(task: VideoTask) -> str:
    data = api_json("/download", {"public_key": task.public_url, "path": task.remote_path})
    href = data.get("href")
    if not href:
        raise RuntimeError(f"No download href returned for {task.video_path}: {data}")
    return href


def http_download(url: str, task: VideoTask, expected_size: int | None, progress: Progress) -> None:
    part = task.dest.with_name(task.dest.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    mode = "ab" if existing else "wb"

    headers = {"User-Agent": "rascar-box-downloader/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        if existing and response.status != 206:
            existing = 0
            mode = "wb"
        downloaded = existing
        with part.open(mode) as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                with progress.lock:
                    progress.active[task.video_path] = downloaded

    final_size = part.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {task.video_path}: got {final_size}, expected {expected_size}"
        )
    part.replace(task.dest)


def download_one(task: VideoTask, expected_size: int | None, progress: Progress) -> str:
    task.dest.parent.mkdir(parents=True, exist_ok=True)

    if task.dest.exists() and expected_size is not None and task.dest.stat().st_size == expected_size:
        with progress.lock:
            progress.skipped += 1
            progress.done_bytes += expected_size
        return f"skip {task.video_path}"

    if task.dest.exists() and expected_size is None:
        with progress.lock:
            progress.skipped += 1
        return f"skip {task.video_path}"

    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            href = get_download_href(task)
            http_download(href, task, expected_size, progress)
            size = task.dest.stat().st_size
            with progress.lock:
                progress.completed += 1
                progress.done_bytes += size
                progress.active.pop(task.video_path, None)
            return f"done {task.video_path} ({fmt_bytes(size)})"
        except Exception as exc:  # noqa: BLE001 - top-level retry boundary.
            last_error = exc
            with progress.lock:
                progress.active.pop(task.video_path, None)
            if attempt == 5:
                break
            time.sleep(min(2**attempt, 30))

    with progress.lock:
        progress.failed += 1
    raise RuntimeError(f"Failed {task.video_path}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DOWNLOAD_WORKERS", "3")))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_paths = [args.data_root / "train/videos.csv", args.data_root / "test/videos.csv"]
    tasks = collect_tasks(csv_paths, args.data_root)

    print(f"Found {len(tasks)} unique videos in CSV files.", flush=True)
    print("Resolving Yandex metadata...", flush=True)
    sizes = resolve_sizes(tasks)
    total = 0
    for index, task in enumerate(tasks, 1):
        size = sizes[task.video_path]
        if size is not None:
            total += size
        print(
            f"[{index:02d}/{len(tasks):02d}] {task.video_path} "
            f"{fmt_bytes(size) if size is not None else 'unknown size'}",
            flush=True,
        )

    existing_ok = sum(
        1
        for task in tasks
        if task.dest.exists()
        and sizes[task.video_path] is not None
        and task.dest.stat().st_size == sizes[task.video_path]
    )
    missing_bytes = sum(
        sizes[task.video_path] or 0
        for task in tasks
        if not (
            task.dest.exists()
            and sizes[task.video_path] is not None
            and task.dest.stat().st_size == sizes[task.video_path]
        )
    )
    print(
        f"Total: {fmt_bytes(total)}; already present: {existing_ok}; "
        f"to download: {len(tasks) - existing_ok} files / {fmt_bytes(missing_bytes)}",
        flush=True,
    )
    if args.dry_run:
        return 0

    progress = Progress()
    progress.total_bytes = total
    monitor = threading.Thread(target=progress.monitor, args=(len(tasks),), daemon=True)
    monitor.start()

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        future_map = {
            pool.submit(download_one, task, sizes[task.video_path], progress): task for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001 - report all failures.
                failures.append(task.video_path)
                print(str(exc), file=sys.stderr, flush=True)

    with progress.lock:
        progress.stop = True
    monitor.join(timeout=1)

    if failures:
        print("Failed files:", file=sys.stderr)
        for path in failures:
            print(f"  {path}", file=sys.stderr)
        return 1

    print("Download complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
