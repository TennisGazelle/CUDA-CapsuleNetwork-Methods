# Build system

Contributor map for configuring and building this repository. Algorithm and kernel
semantics live in [`ARCHITECTURE.md`](ARCHITECTURE.md) and
[`CUDA_ARCHITECTURE.md`](CUDA_ARCHITECTURE.md). CI job coverage is in
[`CICD.md`](CICD.md).

Back to the [docs hub](README.md).

## Developer interface

Prefer the Makefile as the stable entry point. CMake is the build truth.

| Command | Meaning |
|---|---|
| `make help` | List revival targets |
| `make configure-host` | Configure the host-only CMake preset |
| `make unit-test` | Build and run host-only correctness tests |
| `make ctest` | Run CTest against the host CMake build |
| `make docs-check` | Validate required docs and relative Markdown links |
| `make package` / `make package-check` | Create and verify release archives |
| `make ci` | Host PR validation: unit tests + docs + package check |
| `make cuda-compile` | Configure/build CUDA targets (compile only) |
| `make gpu-test` | Run CUDA runtime parity tests (requires a GPU) |
| `make asan-test` | Host tests under AddressSanitizer |
| `make clean` / `make clean-testing-artifacts` | Remove build/release or test-artifact outputs |

## Dependencies

### Host-only (required for `make ci`)

- C++11-capable `g++` or Clang
- CMake ≥ 3.18
- Armadillo (`libarmadillo-dev` on Ubuntu)
- Python 3 (docs check and unit artifact runner)
- Vendored doctest header under `third_party/doctest/` (no network fetch at build time)

```bash
sudo apt-get update
sudo apt-get install -y g++ cmake libarmadillo-dev python3
```

### CUDA (optional; for compile/runtime)

- NVIDIA CUDA toolkit with `nvcc`
- Compatible GPU driver for runtime parity tests
- Armadillo as above

PostgreSQL / libpqxx are **not** required for host tests or basic Capsule Network
inference builds. Enable them only with `CAPSNET_BUILD_GA_DB=ON`.

```bash
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc
```

## CMake options

| Option | Default | Role |
|---|---|---|
| `CAPSNET_BUILD_TESTING` | `ON` | Host unit tests |
| `CAPSNET_BUILD_CUDA` | `OFF` | Enable CUDA language and kernel library |
| `CAPSNET_BUILD_CUDA_APP` | `OFF` | Build the historical `NeuralNets` executable |
| `CAPSNET_BUILD_GA` | `ON` | Include GA / NSGA-II host core |
| `CAPSNET_BUILD_GA_DB` | `OFF` | PostgreSQL / libpqxx fitness cache |
| `CAPSNET_BUILD_CUDA_TESTS` | `OFF` | CUDA primitive parity tests |
| `CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR` | `OFF` | Compatibility switches for historical semantics |
| `CAPSNET_ENABLE_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings as errors |

CUDA architectures are controlled by `CMAKE_CUDA_ARCHITECTURES`. The historical
build hard-coded `-arch sm_30`, which modern NVCC no longer supports. Use a
contemporary architecture for local work (for example `75`, `80`, `86`, or
`native`) and document the exact value in any benchmark or parity report.

## Presets

[`CMakePresets.json`](../CMakePresets.json) defines:

| Preset | Intent |
|---|---|
| `host-debug` | Host tests, Debug, no CUDA |
| `host-relwithdebinfo` | Host tests, RelWithDebInfo, no CUDA |
| `ci-host` | Stable GitHub-hosted CI configure |
| `cuda-compile` | CUDA enabled for compile-only checks |
| `cuda-gpu` | CUDA plus runtime parity tests |

Example host configure:

```bash
cmake --preset ci-host
cmake --build --preset ci-host
ctest --preset ci-host --output-on-failure
```

Example CUDA compile-only configure:

```bash
cmake --preset cuda-compile
cmake --build --preset cuda-compile
```

## Target layout

| Target | Contents |
|---|---|
| `capsnet_host_core` | Utils, MNIST IO, backprop helpers, GA core, simple models |
| `capsnet_cuda_kernels` | CUDA utilities and `CUUnifiedBlob` kernels |
| `NeuralNets` | Historical main executable (when CUDA app is enabled) |
| Host test binaries | Discovered by CTest when testing is enabled |

## Build directories

| Path | Role |
|---|---|
| `.build/` | Legacy Makefile host-test binaries (still used by `make unit-test`) |
| `build/host/` | CMake host presets |
| `build/cuda/` | CMake CUDA presets |
| `testing_artifacts/` | Timestamped unit-test logs / JUnit / metadata |
| `release-assets/` | Packaged source archives and checksums |

## Historical toolchain caveat

The 2018 tree used CMake 2.6-style `FindCUDA`, `cuda_add_executable`, C++11,
Armadillo, pthreads, optional PostgreSQL, Unified Memory, and `sm_30`. A modern
CUDA installation must not be expected to build that tree unchanged. Build
modernization must not silently change capsule math. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## MNIST data location

The reader checks `CAPSNET_DATA_DIR`, then `data/`, then `../data/` for the four
standard IDX files. An explicitly configured directory must contain all four
files; it does not silently fall back. See the root [`README.md`](../README.md).
