#!/usr/bin/env python3
"""Bump a simple x.y.z VERSION file. Mirrors the label-driven HexNets release flow."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_version(text: str) -> tuple[int, int, int]:
    parts = text.strip().split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"expected semantic version x.y.z, got {text!r}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default="VERSION")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--major", action="store_true")
    group.add_argument("--minor", action="store_true")
    group.add_argument("--patch", action="store_true")
    args = parser.parse_args()

    path = Path(args.file)
    major, minor, patch = parse_version(path.read_text(encoding="utf-8"))

    if args.major:
        major, minor, patch = major + 1, 0, 0
    elif args.minor:
        minor, patch = minor + 1, 0
    elif args.patch:
        patch += 1

    version = f"{major}.{minor}.{patch}"
    if args.major or args.minor or args.patch:
        path.write_text(version + "\n", encoding="utf-8")

    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
