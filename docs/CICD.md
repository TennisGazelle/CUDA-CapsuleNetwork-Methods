# CI / CD

Contributor-facing map of GitHub Actions, local validation commands, and release
artifacts. Deep test design lives in [`TESTING.md`](TESTING.md). Build presets
live in [`BUILD.md`](BUILD.md).

Back to the [docs hub](README.md).

## File map

| Path | Role | Change here when… |
|---|---|---|
| [`.github/workflows/pr.yaml`](../.github/workflows/pr.yaml) | PR / revival-branch validation | PR check steps or job matrix |
| [`.github/workflows/cicd.yaml`](../.github/workflows/cicd.yaml) | Label-driven release path | Version bump / release packaging |
| [`.github/actions/`](../.github/actions/) | Reusable composite steps | Shared apt/CMake/test setup |
| [`Makefile`](../Makefile) | Local UX for CI bundles | Developer-facing command names |
| [`scripts/package_release.sh`](../scripts/package_release.sh) | Source archives + checksums | Packaging layout |
| [`scripts/check_docs.py`](../scripts/check_docs.py) | Docs integrity gate | Required spoke list |
| [`VERSION`](../VERSION) | Semver source | Release numbering |

## What each job proves

| Job | Runner | Proves | Does **not** prove |
|---|---|---|---|
| `host-validation` | `ubuntu-latest` | Host tests, docs links, package preview | CUDA compile or GPU runtime |
| `host-sanitizer` | `ubuntu-latest` | Host ASan/UBSan cleanliness where enabled | CUDA correctness |
| `cuda-compile` | CUDA container / CUDA runner | Sources compile under modern NVCC | Kernel numerical parity |

The CUDA compile job installs `git` inside `nvidia/cuda:*-devel` before
`actions/checkout` so submodule-capable clone works (the stock image has no git).
| `gpu-parity` | self-hosted / manual GPU | CPU/CUDA primitive agreement | Published experiment reproduction |

Standard GitHub-hosted runners do not provide an NVIDIA GPU. A green host job
must never be described as "CUDA runtime tested."

## Local equivalents

```bash
make host-ci            # host-validation bundle
make host-asan          # host-sanitizer
make cuda-compile       # cuda-compile (needs toolkit)
make cuda-test          # gpu-parity (needs GPU)
```

Run `make cuda-check` first to verify your CUDA toolkit is installed correctly.

### All targets

| Target | Purpose |
|--------|---------|
| `host-debug` | Build debug configuration (no tests) |
| `host-debug-test` | Run tests against debug build |
| `host-test` | Quick unit tests (no CMake) |
| `host-ci` | Full CI bundle: tests + docs + package |
| `host-asan` | AddressSanitizer build + test |
| `cuda-check` | Verify CUDA toolkit, report missing deps |
| `cuda-compile` | Compile CUDA (toolkit only, no GPU) |
| `cuda-test` | CUDA GPU parity tests (requires GPU) |

### VSCode launch configurations

`.vscode/launch.json` provides IDE buttons for all Makefile targets. The Makefile
is the source of truth; launch.json just invokes it.

## Artifacts

### PR uploads

- Source-package preview under `release-assets/`
- Unit-test artifacts under `testing_artifacts/unit/**` when the CMake/`./unit`
  path is used

Retention is short (on the order of days). Download from the workflow run's
Artifacts UI when investigating a remote failure.

### Release uploads

A merged PR to the maintained base labeled `major`, `minor`, or `patch` produces:

- version bump on `VERSION`
- tag `vX.Y.Z`
- source `tar.gz` / `zip`
- SHA256 checksum manifest
- critical documentation attachments

An unlabeled merge does not cut a release. Manual `workflow_dispatch` on the
CICD workflow validates without releasing.

## Badge / coverage language

Prefer explicit wording:

> Host unit/docs/package validation. CUDA compile and GPU parity are separate
> jobs and may be optional.

Do not imply that GitHub-hosted CI executed kernels.
