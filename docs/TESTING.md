# Testing

Companion to [`BUILD.md`](BUILD.md) and [`CICD.md`](CICD.md). Covers host
correctness tests, CTest/doctest discovery, CUDA primitive parity design, and
tolerance conventions.

Back to the [docs hub](README.md). Colocated runner notes:
[`tests/README.md`](../tests/README.md).

## Tiers

| Tier | Requires | Command | Proves |
|---|---|---|---|
| Host unit | C++ compiler, Armadillo | `make unit-test` or `./unit` | Corrected host math / GA / MNIST IO |
| Docs | Python 3 | `make docs-check` | Required docs + relative links |
| Host sanitizer | same as host + ASan | `make asan-test` | Memory errors in host paths |
| CUDA compile | CUDA toolkit | `make cuda-compile` | Sources compile; **not** runtime |
| GPU parity | NVIDIA GPU + toolkit | `make gpu-test` | CPU/CUDA numerical agreement |

A green host CI badge is **not** GPU runtime validation. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §12.

## Host tests

Current assert-based and doctest-based host coverage includes:

- `Utils` norms, RNG, weight init, flatten/reshape
- NSGA-II core semantics and generation/truncation
- Sequential backprop accumulation helpers
- MNIST IDX validation

Preferred local entry points:

```bash
make unit-test          # Makefile path (CI-stable)
./unit                  # CMake/CTest path with artifacts
./unit -R utils         # filter discovered CTest cases
./unit --list           # list discovered tests
```

`./unit` writes under `testing_artifacts/unit/<timestamp>/`:

- `unit.log`
- `junit.xml` (when CTest supports it)
- `metadata.json` (branch, commit, dirty flag, command, preset, exit code)

## CUDA parity design

Before changing kernels, add a parity test for the primitive. Pattern:

1. tiny deterministic input;
2. independent CPU/reference calculation in the test tree;
3. invoke the existing `CUUnifiedBlob::CUDA_*` wrapper;
4. synchronize and check CUDA errors;
5. compare every output element with a documented tolerance;
6. include zero, small, large, and non-square dimensions where legal;
7. skip cleanly (do not fail) when no GPU runtime is available.

Suggested first primitives (see [`PLAN.md`](../PLAN.md) M3):

1. matrix-vector vote transform
2. routing softmax
3. weighted vote reduction
4. vector squash
5. agreement dot product

Longer layout and kernel inventory: [`CUDA_ARCHITECTURE.md`](CUDA_ARCHITECTURE.md).

## Tolerance policy

| Operation class | Suggested absolute tolerance | Notes |
|---|---:|---|
| Exact integer / indexing | `0` | Shape and discrete results |
| Norms / squash on tiny fixtures | `1e-10` to `1e-8` | `double` host paths |
| Softmax / reductions | `1e-8` to `1e-6` | Shared-memory reductions may differ in order |
| Full routing iteration | document per test | Accumulate only after primitive coverage |

Prefer named helpers in `tests/test_helpers/` over hard-coding magic numbers in
every file. When a historical path intentionally differs from corrected math,
record both behaviors and gate with an explicit compatibility switch.

## Evaluation purity

A corrected `evaluate()` must leave weights, deltas, velocity, and convolutional
filter state unchanged. Characterization of historical `tally(false)` mutation
belongs in tests before any API change. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Adding a host test

1. Prefer a `TEST_CASE` in a doctest-linked binary when CMake testing is enabled.
2. Keep fixtures tiny and deterministic; call `Utils::setRandomSeed` when RNG is involved.
3. Use standard exceptions for invalid input; do not rely on `assert` for production validation.
4. Update [`tests/README.md`](../tests/README.md) with a one-line description.
5. Ensure `make ci` still passes on a GPU-free host.

## Adding a CUDA parity test

1. Put CPU oracles under `tests/` or `tests/test_helpers/`, not in production kernels.
2. Guard runtime with a GPU availability check that skips rather than fails.
3. Document logical dimensions (`I`, `O`, `K`, `T`) next to the fixture.
4. Sync docs in [`CUDA_ARCHITECTURE.md`](CUDA_ARCHITECTURE.md) if indexing contracts change.
5. Do not optimize the kernel in the same change that introduces the parity test.

## Debug / feature flags

Prefer CMake options that feed `CAPSNET_*` compile definitions over scattered
`#define` or `#if 0` blocks. Document new flags here and in [`BUILD.md`](BUILD.md).
