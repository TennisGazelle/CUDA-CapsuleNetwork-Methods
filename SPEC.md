# Restoration specification

## 1. Purpose

Restore the `CUDAify` Capsule Network research codebase into a repository that is understandable, testable, reproducible, and safe for modern agents/contributors to extend, **without erasing its value as a 2018 historical research artifact**.

This specification describes the desired end state. [`PLAN.md`](PLAN.md) decomposes it into work packages.

## 2. Non-goals of the preservation phase

The initial restoration does **not** require:

- replacing the implementation with PyTorch;
- replacing custom kernels with Triton/cuBLAS/cuDNN;
- changing dynamic routing to attention;
- changing the mathematical model to a newer Capsule Network variant;
- producing state-of-the-art accuracy;
- merging every historical branch;
- deleting old data/weights merely because they are untidy.

Those may become later research/optimization projects after reproduction and parity exist.

## 3. Historical truth requirements

The repository SHALL:

1. identify `CUDAify` as the surviving thesis-era CUDA lineage;
2. preserve `master`, `smallNorbReader`, and `hpcvis3-leftover` branch context in documentation;
3. distinguish historical behavior from corrected behavior;
4. retain published historical measurements with their original methodology caveats;
5. never overwrite a historical result with a modern measurement under the same label;
6. provide an experiment provenance table mapping published figures/tables to commits and invocation/configuration when reconstructed.

## 4. Documentation requirements

The repository SHALL provide:

- a useful root README;
- human/research documentation under `docs/`;
- agent onboarding/routing consistent with other TennisGazelle repositories;
- `ai-skills` pinned as a submodule;
- a known-issues ledger;
- a reproducibility policy;
- architecture and CUDA memory/kernel maps;
- a rich explanation of Capsule Networks and why the architecture did not become dominant;
- an independently attackable work plan.

Documentation SHALL use a hub-and-spoke model: root hubs navigate; deep implementation/research context lives in dedicated spokes.

## 5. Build requirements

The corrected/modern build SHALL eventually:

- use contemporary CMake CUDA-language support rather than deprecated `FindCUDA`;
- make CUDA architecture configurable;
- document the oldest/newest supported toolchain;
- make optional dependencies explicit;
- avoid requiring PostgreSQL for basic Capsule Network unit tests/inference;
- expose host-only test targets separately from GPU integration targets;
- provide deterministic build commands suitable for CI.

A historical build recipe MAY use a pinned legacy container/toolchain when necessary.

## 6. Testing requirements

### Host unit tests

Every PR SHALL be able to run meaningful host-only tests without an NVIDIA GPU.

Initial host coverage SHALL include at least:

- `Utils` math/shape helpers;
- known historical numerical semantics such as epsilon handling;
- chromosome decoding or an independent equivalent fixture once decoupled enough to test;
- documentation link integrity.

### CPU/GPU primitive parity

On a GPU-capable environment, corrected mode SHALL compare CPU/reference and CUDA versions of:

- matrix-vector transformations;
- routing softmax;
- weighted vote reduction;
- squash;
- agreement dot product;
- margin loss/gradient;
- error decomposition;
- transposed matrix-vector error propagation;
- class/error reductions;
- transformation-matrix gradient accumulation;
- momentum updates;
- convolution forward/backward;
- tensor/capsule remapping.

### Integration tests

The suite SHALL eventually include:

- one full deterministic routing pass;
- one full deterministic forward pass;
- one backward pass;
- one weight update;
- tiny-batch training;
- pure evaluation that provably does not mutate learned state;
- known synthetic NSGA-II Pareto-front fixtures.

## 7. Evaluation requirements

A corrected API SHALL separate:

- `train` / backpropagation / update;
- `evaluate` / metrics / no mutation;
- `benchmark` / timing with explicit inclusion boundaries.

Evaluation on validation/test data SHALL NOT update weights or optimizer state.

Historical compatibility mode MAY reproduce historical mutation behavior only when explicitly selected and documented.

## 8. Randomness requirements

Corrected mode SHALL provide one reproducibility seed that controls every deterministic-capable RNG source used by:

- weight initialization;
- GA population generation;
- selection/crossover/mutation;
- any dataset shuffling;
- CPU reference stochastic behavior.

Random distributions SHALL honor per-call bounds rather than accidentally capturing first-call parameters.

## 9. Benchmark requirements

Every reported benchmark SHALL state:

- benchmark category (kernel/layer/example/throughput/end-to-end);
- hardware;
- software/toolchain;
- compiler flags;
- CUDA architecture;
- precision;
- workload/configuration;
- warmup/repetition method;
- synchronization boundaries;
- whether allocation/migration/data loading are included.

CPU-vs-GPU comparisons SHALL use intentionally comparable optimization settings.

Modern study SHOULD include optimized library/framework baselines where appropriate.

## 10. CUDA modernization requirements

Modern optimization SHALL be gated by parity tests.

A kernel/memory optimization SHALL:

1. preserve a reference implementation or test oracle;
2. demonstrate numerical equivalence within documented tolerance;
3. benchmark with a stable harness;
4. document changed synchronization/memory semantics.

Potential future techniques include stream overlap, explicit/prefetched memory, operation fusion, cuBLAS/cuDNN, CUDA Graphs, mixed precision, CUTLASS/Triton, or sparse routing. None are required in the preservation PR.

## 11. Data requirements

The repository SHALL eventually classify bundled files as:

- required source fixture;
- externally sourced dataset;
- generated artifact;
- archival experiment result.

External datasets SHOULD have source URLs and checksums. Historical data MAY remain in an archival release even if removed from normal source packages later.

## 12. GA/experiment-cache requirements

The NSGA-II implementation SHALL receive synthetic correctness tests.

A modern fitness cache SHALL include an experiment fingerprint beyond chromosome alone, sufficient to prevent reuse across incompatible:

- code versions;
- datasets;
- seeds;
- evaluation semantics;
- training horizons.

PostgreSQL SHALL be optional for core inference/unit-test builds.

## 13. CI requirements

PR CI SHALL:

- initialize submodules;
- run host unit tests;
- run documentation/link checks;
- validate packaging;
- clearly label CUDA compile/runtime coverage.

GPU runtime CI SHOULD run on a self-hosted or otherwise GPU-capable runner when available.

## 14. Release requirements

Following the established HexNets convention:

- merge to the maintained base branch with `major`, `minor`, or `patch` label -> version bump/tag/release;
- merge without a release label -> no release;
- release contains a packaged source archive and critical documentation;
- release notes are generated by GitHub;
- version is stored in a simple repository file until a compiled/package version API is designed.

## 15. Definition of restored

The preservation/restoration phase is complete when:

- a modern contributor can clone and understand the project without branch archaeology;
- host CI is green and meaningful;
- a documented GPU environment can build/run parity tests;
- historical experiment provenance is reconstructed sufficiently to explain published figures;
- evaluation is pure in corrected mode;
- major suspected numerical defects are either fixed with compatibility notes or explicitly disproven;
- historical and modern benchmark tracks are clearly separated;
- another agent can pick any remaining modernization item from `PLAN.md` without requiring undocumented oral history.
