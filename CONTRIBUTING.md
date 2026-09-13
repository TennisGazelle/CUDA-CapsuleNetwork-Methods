# Contributing

This is a restoration of historical research software. A contribution is successful when it improves the repository **without making it harder to distinguish 2018 behavior from modern corrections**.

## Setup

Clone submodules:

```bash
git submodule update --init --recursive
```

Read:

1. [`AGENTS.md`](AGENTS.md)
2. [`SPEC.md`](SPEC.md)
3. [`PLAN.md`](PLAN.md)
4. [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)
5. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

## Change categories

Please describe a PR as primarily one of:

- **preservation/docs**: improves discoverability/history without changing math;
- **characterization test**: captures current behavior, even if that behavior is suspected wrong;
- **correctness fix**: changes numerical/evaluation behavior with a test and historical note;
- **build/tooling**: updates toolchain/CI/package behavior;
- **performance**: changes execution while preserving tested semantics;
- **research**: intentionally changes model/routing/training semantics.

## Correctness fixes

For a suspected historical defect:

1. add a minimal test demonstrating current behavior;
2. document the issue in `docs/KNOWN_ISSUES.md`;
3. determine whether it may affect published results;
4. implement the correction separately;
5. update reproduction docs;
6. keep compatibility behavior available when needed to reproduce historical results.

## Tests

Host-only tests should run through:

```bash
make unit-test
```

CMake/CTest with artifacts:

```bash
./unit
make ctest
```

The PR validation bundle is:

```bash
make ci
```

Optional scoped format check for newly modernized files:

```bash
make format-check
```

GPU tests must state the GPU/toolkit used. A CUDA compile without a GPU is not a GPU runtime test.
See [`docs/TESTING.md`](docs/TESTING.md) and [`docs/CICD.md`](docs/CICD.md).

## C++ / CUDA conventions

Inspired by long-lived numerical C++ projects (for example Rose process habits),
adapted to this historical CapsNet codebase:

- Preserve C++11 compatibility unless a build-modernization change deliberately raises the standard for a target.
- Use four-space indentation and place opening braces on the declaration line.
- Prefer `#pragma once` for **new** headers; do not churn all historical include guards in the same PR as behavior fixes.
- No `using namespace` in headers. Prefer `static_cast` over C-style casts in new code.
- Keep host-testable logic out of `.cu` files when it does not depend on CUDA.
- Prefer deterministic, single-purpose fixtures and standard exceptions for invalid runtime input; do not rely on `assert` for production validation.
- Document public contracts with `///` in headers: shapes, ownership, synchronization, and whether the call mutates learned state.
- Long algorithm explanations belong in `docs/`; headers should link to the relevant spoke.
- Prefix new compile definitions with `CAPSNET_`. Prefer CMake options over scattered `#define` / `#if 0`.
- Do not remove `cudaDeviceSynchronize()` for performance until a test classifies the barrier.
- Follow the naming and include style of the file being edited instead of reformatting unrelated historical code.
- Optional format checks apply to newly touched modernization files; do not mass-format the 2018 tree.

See also [`docs/BUILD.md`](docs/BUILD.md), [`docs/TESTING.md`](docs/TESTING.md), and [`.clang-format`](.clang-format).

## Documentation

Update the relevant spoke in the same PR. Prefer links from README/AGENTS instead of duplicating long explanations.

## Performance claims

Include hardware, compiler flags, CUDA version/architecture, precision, configuration, warmup/repetition, synchronization boundaries, and benchmark category.

Do not compare a modern optimized result with a historical number without explicitly labeling both environments.
