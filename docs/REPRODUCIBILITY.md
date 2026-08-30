# Reproducibility and benchmark policy

The goal of the revival is to make the historical implementation reproducible **without rewriting history**. That requires separating three different targets that are easy to conflate.

## 1. Three reproduction targets

### A. Historical artifact reproduction

Question:

> Can we rebuild and run the code in behavior close to the environment that produced the 2018 work?

This may require:

- an older CUDA toolkit;
- a compatible GPU architecture or container/emulation strategy;
- historical compiler versions/flags;
- Armadillo and libpqxx versions;
- PostgreSQL schema/config;
- original MNIST files;
- original experiment commit and command/function entry point.

Historical reproduction may intentionally retain known bugs if those bugs were present in the published experiment path. Such a run must be labeled historical/compatibility behavior.

### B. Corrected reproduction

Question:

> If we repair confirmed correctness defects while preserving the model definition, what accuracy and performance do we obtain?

This run should use:

- explicit train/evaluate separation;
- deterministic seeds where possible;
- unit-tested math primitives;
- CPU/GPU numerical parity checks;
- documented compiler/toolchain versions.

Corrected results belong beside historical results, not on top of them.

### C. Modern baseline study

Question:

> How does the same or closely related CapsNet idea perform on contemporary hardware/software against optimized alternatives?

This may use:

- current CUDA;
- modern CMake;
- cuBLAS/cuDNN;
- PyTorch or other framework baselines;
- mixed precision where appropriate;
- current GPUs;
- profiling tools;
- contemporary attention/routing alternatives.

This is a new research experiment, not a reproduction of the thesis.

## 2. Historical branch/commit problem

`CUDAify` is the correct broad lineage, but the tip is not automatically the exact code for every figure. The branch accumulated changes through August 2018, including GA, threading/stream experiments, and experiment commands.

Reproduction therefore needs an experiment provenance table:

| Published artifact | Figure/table | commit | entry point | config | dataset | hardware | notes |
|---|---|---|---|---|---|---|---|
| CUDA paper | TBD | TBD | TBD | TBD | MNIST | historical UNR GPU | reconstruct |
| Thesis | performance plots | TBD | TBD | TBD | MNIST | historical | reconstruct |
| Thesis | NSGA-II results | TBD | TBD | TBD | MNIST | historical HPC | reconstruct |

Do not fill unknown cells by guessing.

## 3. Evaluation purity requirement

A modern `evaluate()` operation must be observational with respect to learned model state.

Acceptance test:

```text
snapshot weights + optimizer state
run evaluation over test fixture
snapshot again
assert byte/numerical equality
```

The surviving CUDA `tally(false)` path appears not to satisfy this property, so it is the first high-priority historical audit.

## 4. Determinism

The historical code uses several randomness sources/styles. A reproducibility pass must inventory:

- `rand()` and `srand()` usage;
- `std::random_device`;
- `std::mt19937` initialization;
- static distributions;
- GPU nondeterminism/atomic ordering where applicable;
- GA mutation/crossover selection;
- initial weights.

A single `--seed` or config field should eventually control all deterministic-capable random sources in corrected mode.

Historical mode may preserve original seeding behavior when reconstructing old results.

PR #7 routes the corrected host utilities and GA operators through one shared
`std::mt19937` and exposes `Utils::setRandomSeed` for repeatable tests. This is
an intermediate boundary, not full experiment fingerprinting: the seed is not
yet a top-level CLI/config option, other historical randomness sources still
need inventory, and the shared generator is not a deterministic thread-safe
experiment stream.

## 5. Numerical parity before training parity

Full training curves are a bad first correctness oracle because tiny differences accumulate.

Restoration should compare CPU and GPU from the bottom upward:

1. vector length / normalization;
2. squash;
3. matrix-vector vote transformation;
4. routing softmax;
5. weighted vote reduction;
6. agreement dot products;
7. one complete routing iteration;
8. all routing iterations;
9. margin loss and derivative;
10. error decomposition;
11. transposed matrix propagation;
12. lower-capsule reduction;
13. weight-gradient accumulation;
14. momentum update;
15. convolution forward/backward;
16. one complete example;
17. tiny deterministic batch;
18. tiny training run;
19. full dataset training.

Every level should use tolerances appropriate to the operation and numeric type.

## 6. Benchmark taxonomy

Every benchmark result should identify which category it belongs to.

### Kernel microbenchmark

Times one primitive with fixed input shapes, warmed up, excluding setup/allocation unless setup is the object of study.

### Layer benchmark

Times a complete routing or convolutional layer including required synchronization.

### Example latency

Times one complete example through forward/backward/update as specified.

### Throughput benchmark

Measures examples/second for a fixed batch/workload.

### End-to-end training benchmark

Includes data loading, forward, backward, updates, and whatever evaluation/checkpointing policy is explicitly stated.

Never compare numbers from different categories without saying so.

## 7. Timing hygiene

A modern benchmark harness should record:

- GPU model;
- driver version;
- CUDA toolkit/runtime versions;
- CPU model;
- compiler and version;
- compiler flags;
- GPU architecture flag;
- optimization level;
- numerical precision;
- batch size;
- capsule dimensions;
- tensor-channel count;
- routing iterations;
- warmup count;
- repetition count;
- synchronization boundaries;
- whether allocation/data migration is included;
- median plus dispersion, not only one best sample.

## 8. Historical speedup caveat

The thesis reports very large CUDA-vs-sequential speedups. Those results are evidence that routing exposes substantial parallel work and that the custom memory/kernel strategy succeeded against the reference implementation.

They should not be generalized into claims against optimized GPU libraries/frameworks because:

- the reference implementation is Armadillo/object-oriented research code;
- compiler optimization context was not perfectly symmetric;
- convolution was hand-written rather than cuDNN-backed;
- the comparison predates modern compiler/framework kernels;
- hardware has changed radically.

The correct modern follow-up is to keep the old numbers, reproduce them if possible, and add a new apples-to-apples benchmark table.

## 9. Unified Memory considerations

The CUDA implementation uses `cudaMallocManaged`. Reproduction must clarify whether timings include:

- first-touch page migration;
- host-side debug reads;
- managed-memory faults;
- explicit prefetch (historical code does not appear to rely on a modern prefetch strategy);
- allocation time.

A first run can be dramatically different from a warmed run. Modern profiling should track migration/fault behavior.

## 10. Dataset provenance

Historical MNIST files are bundled in the repository. Future cleanup should replace opaque large binaries with a documented download/checksum process while retaining either:

- checksums of the historical files; or
- an archival release containing the exact experiment inputs.

`smallNorbReader` remains an unmerged branch and should not be implied to be part of the canonical thesis run until branch history says otherwise.

## 11. PostgreSQL/GA reproducibility

The GA evaluation cache introduces external state. Reproducing the search requires documenting:

- schema;
- connection configuration;
- uniqueness key/chromosome encoding;
- which fitness values are stored;
- whether cache values depend on random seeds;
- how an old result is invalidated if evaluation semantics change.

A cache keyed only by chromosome is unsafe if fitness also depends on code version, seed, dataset, or evaluation procedure. Modernized cache keys should include an experiment/version fingerprint.

## 12. CI reality

Standard GitHub-hosted runners do not provide an NVIDIA GPU. The CI design should therefore distinguish:

- host-only unit tests that can run on every PR;
- source/build configuration checks;
- CUDA compilation checks that do not imply runtime validation;
- optional self-hosted GPU integration tests.

A green non-GPU CI badge must not be described as "CUDA runtime tested."

## 13. Release artifacts

Release packaging should preserve source/documentation and pin the `ai-skills` submodule reference. Large generated datasets/weights should eventually have an explicit policy.

Each release should include or generate:

- version/tag;
- source archive;
- README;
- `SPEC.md` / `PLAN.md` at the time of release;
- known issues/reproducibility docs;
- generated GitHub release notes.

The release workflow in this revival mirrors the label-driven pattern used by `TennisGazelle/HexNets`: a merged PR labeled `major`, `minor`, or `patch` produces a version bump, tag, packaged archive, and GitHub Release. An unlabeled merge does not cut a release.
