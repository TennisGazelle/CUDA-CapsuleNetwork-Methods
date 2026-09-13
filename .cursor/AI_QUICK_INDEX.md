# AI Assistant Quick Index

**Human/research docs:** [`README.md`](../README.md) · [`docs/`](../docs/README.md) · [`WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md`](../WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md)

## Read first

| Topic | Link |
|---|---|
| Canonical agent rules | [`rules/README.md`](rules/README.md) |
| Historical preservation | [`rules/historical-preservation.mdc`](rules/historical-preservation.mdc) |
| Documentation sync | [`rules/documentation-sync.mdc`](rules/documentation-sync.mdc) |
| Verify ambiguity | [`rules/implementation-clarity.mdc`](rules/implementation-clarity.mdc) |
| Known defects / audit flags | [`../docs/KNOWN_ISSUES.md`](../docs/KNOWN_ISSUES.md) |
| Reproduction caveats | [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) |
| Target state | [`../SPEC.md`](../SPEC.md) |
| Work packages | [`../PLAN.md`](../PLAN.md) |

## Task router

| If the task touches... | Read | Main code |
|---|---|---|
| Capsule math / routing | [`../docs/CAPSULE_NETWORKS.md`](../docs/CAPSULE_NETWORKS.md), [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | `src/CapsuleNetwork/` |
| CUDA kernels / memory | [`../docs/CUDA_ARCHITECTURE.md`](../docs/CUDA_ARCHITECTURE.md) | `src/models/CUUnifiedBlob.cu`, `src/CapsuleNetwork/CUCapsuleNetwork/` |
| Published results | [`../docs/THESIS.md`](../docs/THESIS.md), [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) | experiment paths in `src/main.cu`, `src/GA/` |
| Genetic search / NSGA-II | [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | `src/GA/` |
| Branch archaeology | [`../docs/REPO_AUDIT.md`](../docs/REPO_AUDIT.md) | Git history / PR #3 / PR #5 |
| Tests | [`../docs/TESTING.md`](../docs/TESTING.md), [`../tests/README.md`](../tests/README.md) | `tests/`, `Makefile`, `./unit` |
| Build / CMake | [`../docs/BUILD.md`](../docs/BUILD.md) | `CMakeLists.txt`, `CMakePresets.json`, `Makefile` |
| CI/release | [`../docs/CICD.md`](../docs/CICD.md), [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) | `.github/workflows/`, `scripts/` |
| Experiment provenance | [`../docs/EXPERIMENT_PROVENANCE.md`](../docs/EXPERIMENT_PROVENANCE.md) | `src/main.cu`, `src/GA/` |
| Modern CapsNet research | [`../WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md`](../WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md) | future work only |

## Branch truth

- `master`: older sequential/reference implementation.
- `CUDAify`: thesis-era CUDA implementation and the base for revival work.
- `smallNorbReader`: open follow-on PR into `CUDAify`; do not silently absorb it.
- `hpcvis3-leftover`: later archaeological fork with a 2020 recovery commit; not automatically newer/better than final `CUDAify`.

## Change discipline

For a historical correctness bug:

1. prove it with a minimal test or trace;
2. record the historical behavior in `docs/KNOWN_ISSUES.md`;
3. identify whether published figures could have used that path;
4. fix it only in a clearly labeled modernization change;
5. retain a reproduction note or compatibility switch if needed.
