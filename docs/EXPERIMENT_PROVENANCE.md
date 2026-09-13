# Experiment provenance

Table for mapping published figures and tables to surviving commits and
invocation paths. **Do not fill unknown cells by guessing.** See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Back to the [docs hub](README.md).

## Status

Reconstruction is incomplete. Empty cells mean evidence has not been located
yet. When a cell is filled, cite the git evidence (commit, branch tip, or
historical script) in the notes column.

## Provenance table

| Published artifact | Figure/table | commit | entry point | config | dataset | hardware | toolchain | notes |
|---|---|---|---|---|---|---|---|---|
| CUDA paper | TBD | TBD | TBD | TBD | MNIST | historical UNR GPU | TBD | reconstruct |
| Thesis | performance plots | TBD | TBD | TBD | MNIST | historical | TBD | reconstruct |
| Thesis | NSGA-II results | TBD | TBD | TBD | MNIST | historical HPC | TBD | reconstruct |

## Candidate entry points to audit

- `src/main.cu` experiment switches and CLI paths
- `CUCapsuleNetwork::train` / `tally` / `runEpoch`
- `src/GA/` real evaluation vs historical `fakeEvaluate` (removed from the live path in PR #7)
- `slurm/` and `cmds/` historical cluster helpers
- `hpcvis3-leftover` recovery branch (mine selectively; do not treat as automatic successor)

## Corrected vs historical labeling

When a modern run is recorded, label it **corrected** or **historical
compatibility** explicitly, and include:

- git SHA and dirty flag
- CMake preset / build type
- compiler and CUDA toolkit versions
- `CMAKE_CUDA_ARCHITECTURES`
- seed / config / dataset hash when available
