#!/usr/bin/env bash
# Scoped format check for newly modernized paths only.
# Usage: scripts/check_format.sh [files...]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format not found; skipping format check" >&2
  exit 0
fi

if [[ $# -gt 0 ]]; then
  files=("$@")
else
  files=(
    tests/test_utils.cpp
    tests/test_eval_no_mutation.cpp
    tests/test_cu_matrix_vector_mult.cpp
    tests/test_helpers/near.hpp
    tests/test_helpers/seed.hpp
    src/CapsuleNetwork/EvalPolicy.cpp
    include/CapsuleNetwork/EvalPolicy.h
  )
fi

existing=()
for f in "${files[@]}"; do
  [[ -f "$f" ]] && existing+=("$f")
done

if [[ ${#existing[@]} -eq 0 ]]; then
  echo "no scoped files to check"
  exit 0
fi

failed=0
for f in "${existing[@]}"; do
  if ! clang-format --dry-run -Werror "$f" >/dev/null 2>&1; then
    echo "format drift: $f" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "reformat with: clang-format -i <files>" >&2
  exit 1
fi

echo "scoped format check passed (${#existing[@]} paths)"
