#!/usr/bin/env python3
"""Small dependency-free documentation integrity check for PR CI."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "AGENTS.md",
    "SPEC.md",
    "PLAN.md",
    "WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md",
    "docs/README.md",
    "docs/ARCHITECTURE.md",
    "docs/CUDA_ARCHITECTURE.md",
    "docs/CAPSULE_NETWORKS.md",
    "docs/THESIS.md",
    "docs/KNOWN_ISSUES.md",
    "docs/REPRODUCIBILITY.md",
    "docs/REPO_AUDIT.md",
    ".cursor/AI_QUICK_INDEX.md",
    ".cursor/rules/README.md",
]

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def check_required() -> list[str]:
    return [f"missing required documentation: {path}" for path in REQUIRED if not (ROOT / path).exists()]


def is_repo_owned_doc(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    ignored_parts = {".git", ".build", "release-assets"}
    if any(part in ignored_parts for part in rel.parts):
        return False

    # ai-skills is a pinned external submodule. Its own docs may intentionally
    # contain illustrative placeholder links such as [text](path); audit the
    # consuming repo's links *to* the submodule, but not the vendored docs
    # inside it.
    if rel.parts[:2] == (".agents", "ai-skills"):
        return False

    return True


def markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not is_repo_owned_doc(path):
            continue
        if path.is_file() and path.suffix.lower() in {".md", ".mdc"}:
            files.append(path)
    return files


def clean_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    # Strip an optional quoted Markdown title after a path.
    target = re.split(r"\s+[\"']", target, maxsplit=1)[0]
    return unquote(target)


def check_links() -> list[str]:
    errors: list[str] = []
    for doc in markdown_files():
        text = doc.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = clean_target(match.group(1))
            if not target or target.startswith("#"):
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue

            path_part = target.split("#", 1)[0].split("?", 1)[0]
            if not path_part:
                continue

            resolved = (doc.parent / path_part).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{doc.relative_to(ROOT)}: link escapes repository: {target}")
                continue

            if not resolved.exists():
                errors.append(f"{doc.relative_to(ROOT)}: broken relative link: {target}")
    return errors


def main() -> int:
    errors = check_required() + check_links()
    if errors:
        print("documentation check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"documentation check passed ({len(markdown_files())} markdown/rule files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
