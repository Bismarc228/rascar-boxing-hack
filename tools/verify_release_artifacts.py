#!/usr/bin/env python3
"""Verify release artifact files against the committed SHA manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("release_artifacts/release_artifacts_manifest.tsv")
STAGES = ("external", "generated_input", "generated_output", "upstream_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--release",
        action="append",
        choices=["all", "shared", "rc1", "rc2"],
        default=None,
        help="Release filter. rc1/rc2 also include shared rows unless --no-shared is set.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=["all", *STAGES],
        default=None,
        help="Artifact stage filter.",
    )
    parser.add_argument("--kind", action="append", help="Optional kind filter, repeatable.")
    parser.add_argument("--no-shared", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on missing or mismatched files.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = ROOT / args.manifest
    rows = select_rows(load_manifest(manifest_path), args)
    if not rows:
        raise SystemExit("no manifest rows selected")

    ok = []
    missing = []
    mismatched = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            missing.append(row)
            continue
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(row["size_bytes"]) or digest != row["sha256"]:
            mismatched.append((row, size, digest))
            continue
        ok.append(row)

    if not args.quiet:
        print(
            "release_artifacts "
            f"manifest={args.manifest} selected={len(rows)} ok={len(ok)} "
            f"missing={len(missing)} mismatched={len(mismatched)}"
        )
        for row in missing:
            print(f"MISSING\t{row['release']}\t{row['stage']}\t{row['kind']}\t{row['path']}\t{row['restore']}")
        for row, size, digest in mismatched:
            print(
                "MISMATCH\t"
                f"{row['release']}\t{row['stage']}\t{row['kind']}\t{row['path']}\t"
                f"expected_size={row['size_bytes']}\tactual_size={size}\t"
                f"expected_sha256={row['sha256']}\tactual_sha256={digest}\t{row['restore']}"
            )

    if args.strict and (missing or mismatched):
        return 1
    return 0


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    releases = set(args.release or ["all"])
    stages = set(args.stage or ["all"])
    kinds = set(args.kind or [])

    selected = []
    seen_paths = set()
    for row in rows:
        if not release_selected(row["release"], releases, args.no_shared):
            continue
        if "all" not in stages and row["stage"] not in stages:
            continue
        if kinds and row["kind"] not in kinds:
            continue
        path = row["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        selected.append(row)
    return selected


def release_selected(release: str, requested: set[str], no_shared: bool) -> bool:
    if "all" in requested:
        return True
    if release in requested:
        return True
    return release == "shared" and not no_shared and bool(requested & {"rc1", "rc2"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
