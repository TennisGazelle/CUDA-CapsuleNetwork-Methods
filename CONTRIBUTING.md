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

The PR validation bundle is:

```bash
make ci
```

GPU tests must state the GPU/toolkit used. A CUDA compile without a GPU is not a GPU runtime test.

## Documentation

Update the relevant spoke in the same PR. Prefer links from README/AGENTS instead of duplicating long explanations.

## Performance claims

Include hardware, compiler flags, CUDA version/architecture, precision, configuration, warmup/repetition, synchronization boundaries, and benchmark category.

Do not compare a modern optimized result with a historical number without explicitly labeling both environments.
