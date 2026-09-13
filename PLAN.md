# Restoration and modernization plan

This plan is intentionally decomposed so another agent can take one milestone without needing the original conversation. Read [`SPEC.md`](SPEC.md), [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), and [`.cursor/AI_QUICK_INDEX.md`](.cursor/AI_QUICK_INDEX.md) first.

Status legend: `[ ]` not started, `[~]` partial, `[x]` complete.

---

## M0 — Historical preservation and repository orientation

**Goal:** make the surviving artifact legible before behavior changes.

**Status:** `[~]` foundation implemented in revival PR.

### Deliverables

- [x] identify `CUDAify` as thesis-era base;
- [x] document open PR #3 and PR #5 lineage;
- [x] document `hpcvis3-leftover` as a fork/recovery branch;
- [x] revitalize README;
- [x] add research documentation hub;
- [x] add agent hub/rules/task router;
- [x] pin `ai-skills` submodule and harness links;
- [x] add known-issues/reproducibility ledgers;
- [ ] reconstruct exact commit/entry point for every published timing figure;
- [ ] reconstruct exact commit/entry point for thesis accuracy/GA plots;
- [x] create `docs/EXPERIMENT_PROVENANCE.md` table once evidence is available.

### Acceptance criteria

No contributor needs to discover the thesis implementation by guessing branches. Unknown experiment provenance is explicit rather than inferred.

---

## M1 — Host-only correctness baseline

**Goal:** make every PR run meaningful tests without requiring CUDA hardware.

**Status:** `[x]` completed in revival PR #7.

### Deliverables

- [x] add initial `Utils` unit-test executable;
- [x] add Makefile targets for host unit tests and doc checks;
- [x] add PR workflow;
- [x] expand `Utils` tests around edge cases;
- [x] characterize and correct `getWeightRand` and varying-bound `getRandBetween(double)`;
- [x] characterize and correct double-epsilon norm semantics;
- [x] test `getAsOneDim`/`asCapsuleVectors` round trips including malformed shapes;
- [x] decouple chromosome decoding enough for host-only tests;
- [x] add synthetic Pareto-front and generation/truncation tests for NSGA-II helpers.

### Files

- `tests/`
- `src/Utils.cpp`, `include/Utils.h`
- `src/GA/`, `include/GA/`
- `Makefile`
- `.github/workflows/pr.yaml`

### Acceptance criteria

`make ci` succeeds on a clean Ubuntu host with documented apt dependencies and catches a deliberately broken utility invariant.

---

## M2 — Modern build-system compatibility without math changes

**Goal:** configure/build the historical source on a contemporary CUDA toolchain while preserving numerical algorithms.

**Status:** `[~]` Rose-inspired CMake presets / host-CUDA split in progress.

### Deliverables

- [x] raise CMake minimum to a contemporary supported version (3.18+);
- [x] enable CUDA as a first-class CMake language behind an option;
- [x] replace deprecated `FindCUDA`/`cuda_add_executable`;
- [x] make CUDA architectures configurable;
- [x] make PQXX/GA database support optional;
- [x] add `CAPSNET_BUILD_TESTING`, `CAPSNET_BUILD_CUDA`, `CAPSNET_BUILD_GA`, and related options;
- [x] add `CMakePresets.json` (`ci-host`, `host-debug`, `cuda-compile`, `cuda-gpu`);
- [x] document build map in `docs/BUILD.md`;
- [ ] create a documented legacy container/recipe if modern compile requires material compatibility changes;
- [ ] verify full `NeuralNets` link on a contemporary CUDA host and record the toolkit matrix.

### Gotchas

- `sm_30` is unsupported by current toolchains;
- old source may rely on deprecated CUDA APIs/compiler permissiveness;
- changing default C++ standard can expose latent bugs;
- do not opportunistically change capsule math in this milestone.

### Acceptance criteria

A documented current CUDA toolkit can compile the main CUDA targets. Host tests remain buildable without CUDA/PQXX.


---

## M3 — CUDA primitive parity harness

**Goal:** establish a numerical truth boundary before optimization.

### Work packages

Each item can be a separate PR:

- [x] M3.1 matrix-vector vote transform;
- [ ] M3.2 routing softmax;
- [ ] M3.3 weighted vote reduction;
- [ ] M3.4 vector squash;
- [ ] M3.5 agreement dot product;
- [ ] M3.6 margin loss + gradient;
- [ ] M3.7 scaled error decomposition;
- [ ] M3.8 transposed matrix-vector propagation;
- [ ] M3.9 multi-vector error reduction;
- [ ] M3.10 transformation-matrix gradient accumulation;
- [ ] M3.11 momentum update;
- [ ] M3.12 tensor/capsule remapping;
- [ ] M3.13 convolution forward;
- [ ] M3.14 convolution backward.

### Test design

For each primitive:

1. tiny deterministic input;
2. independent CPU/reference calculation;
3. CUDA invocation;
4. synchronize;
5. compare every output element with documented tolerance;
6. include zero, small, large, and non-square dimensions where legal.

### Acceptance criteria

Every CUDA primitive used by full forward/backward has direct parity coverage.

---

## M4 — Full-network parity and confirmed bug triage

**Goal:** prove where sequential and CUDA paths agree/disagree.

### Deliverables

- [ ] deterministic shared weight initialization/copy;
- [ ] one routing-iteration parity test;
- [ ] three-routing-iteration parity test;
- [ ] full forward pass parity;
- [ ] loss parity;
- [ ] full backward error parity;
- [ ] weight-gradient parity;
- [ ] post-update weight parity;
- [x] investigate and correct suspicious sequential `primaryCapsError` indexing;
- [x] investigate and correct historical random initialization bugs;
- [x] record the confirmed fixes with compatibility/reproduction notes;

### Acceptance criteria

Differences are either below tolerance or have a documented, tested explanation.

---

## M5 — Evaluation/train separation and dataset integrity

**Goal:** eliminate accidental leakage in corrected mode and make data boundaries explicit.

**Status:** `[~]` corrected `evaluate()` API added; GPU weight-snapshot proof still open.

### Deliverables

- [x] introduce pure `evaluate` path;
- [x] host policy tests for historical vs corrected mutation contracts;
- [ ] GPU test that evaluation cannot mutate weights/velocity/deltas;
- [x] separate metric accumulation from backprop in corrected `evaluate`;
- [ ] reconstruct whether published experiments used mutating `tally(false)`;
- [ ] add validation-set concept if needed for hyperparameter/GA fitness;
- [ ] checksum/document historical MNIST inputs;
- [ ] reconcile `smallNorbReader` only through a dedicated, tested PR;
- [ ] generalize image shapes only after preserving MNIST behavior.

### Acceptance criteria

Test labels never update learned state in corrected mode; historical reproduction behavior is separately selectable/documented if required.

---

## M6 — Randomness and experiment fingerprinting

**Goal:** make repeated corrected experiments explainably reproducible.

### Deliverables

- [ ] one top-level seed;
- [x] eliminate first-call-bound static distribution bug;
- [x] fix/replace weight initializer with documented distribution;
- [ ] seed GA operations deterministically;
- [ ] record config + seed + git SHA + dataset hash for each run;
- [ ] add experiment fingerprint to database cache key;
- [ ] document unavoidable GPU nondeterminism.

### Acceptance criteria

Two corrected tiny runs with the same seed/config produce the same expected metrics within documented deterministic constraints.

---

## M7 — Historical experiment reproduction

**Goal:** rerun the thesis/paper experiments with provenance.

### Deliverables

- [ ] historical toolchain recipe;
- [ ] CUDA paper timing configurations;
- [ ] thesis timing configurations;
- [ ] sequential and CUDA exact compiler flags;
- [ ] original hardware metadata reconstructed where possible;
- [ ] published plot data regenerated or compared against archived data;
- [ ] NSGA-II experiment configuration reconstructed;
- [ ] result report distinguishing reproduced / approximately reproduced / unreproducible.

### Acceptance criteria

Every major published number has provenance and a reproduction status.

---

## M8 — CI and release hardening

**Goal:** turn the initial workflow scaffold into trustworthy continuous verification.

**Status:** `[~]` tiered jobs and checksum packaging in progress.

### Deliverables

- [x] host PR validation scaffold;
- [x] label-driven release packaging scaffold;
- [x] `docs/CICD.md` contributor map;
- [x] modern CUDA compile job (container/compile-only);
- [x] optional/self-hosted GPU runtime job skeleton;
- [x] sanitizer job for host code;
- [x] release manifest/checksum generation;
- [ ] cache dependencies/build outputs appropriately;
- [ ] automatic documentation of tested CUDA/toolchain matrix from real GPU runs.

### Acceptance criteria

CI makes it impossible to confuse host test success, CUDA compile success, and GPU runtime success.

---

## M9 — Modern performance baseline

**Goal:** rerun the systems question fairly on contemporary hardware.

### Deliverables

- [ ] optimized CPU baseline (`-O3`, appropriate vectorization, stable threading policy);
- [ ] historical custom CUDA baseline;
- [ ] modernized custom CUDA baseline;
- [ ] cuBLAS/cuDNN-backed operations where appropriate;
- [ ] framework implementation baseline;
- [ ] profiler traces showing compute, launch, sync, and managed-memory behavior;
- [ ] kernel/layer/end-to-end benchmark taxonomy;
- [ ] confidence intervals/median dispersion.

### Acceptance criteria

No speedup claim lacks hardware/software/flags/workload context.

---

## M10 — Modern CUDA optimization

**Goal:** optimize only after M3/M4 correctness gates.

Candidate independent PRs:

- [ ] remove unnecessary synchronization;
- [ ] explicit memory placement/prefetch;
- [ ] fuse routing primitives;
- [ ] batched matrix operations;
- [ ] cuDNN convolution;
- [ ] CUDA Graph capture;
- [ ] stream overlap;
- [ ] mixed precision study;
- [ ] CUTLASS/Triton experiments;
- [ ] multi-GPU partitioning based on thesis future-work sketches.

Every optimization requires parity tests and benchmark evidence.

---

## M11 — Capsule research revival

**Goal:** revisit the representation idea with modern routing/measurement rather than merely porting 2017 architecture.

Potential research tracks:

### M11.A Routing comparison

- [ ] original dynamic routing;
- [ ] attention routing;
- [ ] self-attention routing;
- [ ] sparse/top-k routing;
- [ ] window/local routing;
- [ ] EM/cluster-style routing;
- [ ] adaptive routing iterations.

### M11.B Representation evaluation

Measure whether capsule state actually helps:

- [ ] viewpoint extrapolation;
- [ ] occlusion;
- [ ] compositional generalization;
- [ ] overlapping/multi-object scenes;
- [ ] sample efficiency;
- [ ] latent transformation linearity/equivariance;
- [ ] calibration of vector length as presence evidence.

### M11.C Scaling

- [ ] CIFAR-scale controlled study;
- [ ] larger natural-image benchmark only after computational feasibility;
- [ ] sparse relationship graph;
- [ ] shared/low-rank transformation matrices.

### Acceptance criteria

The research separates the value of **capsule representations** from the value/cost of **original dynamic routing**.

---

## Suggested next PRs

The highest-leverage order after this foundation is:

1. **M2 modern CMake compile-only port**;
2. **M3 first five routing primitive parity tests**;
3. **M5 pure evaluation API**, once historical mutation behavior is pinned by a characterization test;
4. **M7 experiment provenance reconstruction** in parallel with the build work;
5. **M8 sanitizer coverage** for the host parsing and GA ownership boundaries.

Do not start M10/M11 merely because they are more exciting. The parity/reproduction boundary is what will make those experiments scientifically interpretable.
