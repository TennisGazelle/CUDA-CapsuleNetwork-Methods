# Unit tests

Host and CUDA tests live under this directory. Host correctness can run without
an NVIDIA GPU. CUDA parity tests require a GPU and must skip cleanly on
host-only machines.

Deep design notes: [`docs/TESTING.md`](../docs/TESTING.md).

## TL;DR

From the repository root:

```bash
make unit-test
```

CMake/CTest path with timestamped artifacts:

```bash
./unit
./unit --list
./unit -R utils
```

## Current host coverage

| File | Scope |
|---|---|
| `test_utils.cpp` | Norms, RNG, weight init, flatten/reshape (doctest when CMake-built) |
| `test_ga.cpp` | NSGA-II dominance, crowding, sorting, stats |
| `test_ga_generation.cpp` | Offspring generation and truncation invariants |
| `test_backprop.cpp` | Sequential backprop accumulation helpers |
| `test_mnist_io.cpp` | MNIST IDX validation and path resolution |
| `test_cu_matrix_vector_mult.cpp` | CUDA/CPU matrix-vector vote parity (GPU optional) |
| `test_eval_no_mutation.cpp` | Characterization / corrected evaluation purity |

## Helpers

`tests/test_helpers/` provides shared approximate comparisons, seed setup, and
optional CUDA availability checks. Prefer those helpers over duplicating
tolerance constants.

## Outputs

`./unit` writes under `testing_artifacts/unit/<timestamp>/`:

- `unit.log`
- `junit.xml` when supported
- `metadata.json`

Clean with:

```bash
make clean-testing-artifacts
```

## Adding tests

1. Add a new `tests/test_*.cpp` (or `TEST_CASE` in an existing doctest binary).
2. Wire it through CMake when using the CMake path; keep Makefile entries only
   if the host CI still needs the direct `g++` path.
3. Document the one-line scope in this file.
4. Update [`docs/TESTING.md`](../docs/TESTING.md) if the tier or pattern changes.
