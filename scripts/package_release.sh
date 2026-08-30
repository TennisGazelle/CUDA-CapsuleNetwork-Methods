#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="$(tr -d '[:space:]' < VERSION)"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "VERSION must be semantic x.y.z; got '$version'" >&2
  exit 1
fi

name="cuda-capsule-network-methods-${version}"
mkdir -p release-assets
rm -f "release-assets/${name}.tar.gz" "release-assets/${name}.zip"

git archive --format=tar.gz --prefix="${name}/" HEAD > "release-assets/${name}.tar.gz"
git archive --format=zip --prefix="${name}/" HEAD > "release-assets/${name}.zip"

printf 'created:\n  %s\n  %s\n' \
  "release-assets/${name}.tar.gz" \
  "release-assets/${name}.zip"
