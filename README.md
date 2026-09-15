# CUDA Capsule Network Methods

Historical C++/CUDA research implementation of dynamically routed Capsule Networks, including a sequential reference implementation, a custom CUDA execution/memory model, and an NSGA-II search over Capsule Network architecture and loss parameters.

> **Repository status:** this codebase originated in 2017–2018 research and is being restored as a reproducible historical artifact before any aggressive algorithmic modernization. The thesis-era CUDA work lives on the `CUDAify` lineage, not on `master`.

## Research artifacts

- CUDA-course / conference paper: [A GPU Acceleration Method for Dynamically Routed Capsule Networks](https://www.cse.unr.edu/~fredh/papers/conf/194-amlfdrcl/paper.pdf)
- Master's thesis: [Evolving GPU-Accelerated Capsule Networks](https://www.cse.unr.edu/~fredh/papers/thesis/071-lopez/thesis.pdf)
- Original Capsule Network reference: [Sabour, Frosst, and Hinton, Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

See [`docs/THESIS.md`](docs/THESIS.md) for the relationship between the paper, thesis, code, and historical branches.

## What this repository actually explores

The main contribution is more specific than "CapsNet in CUDA." The project asks how a routing-heavy neural architecture whose natural mathematical representation is a hierarchy of vectors and matrices can be transformed into a GPU-friendly representation.

The CUDA implementation flattens capsule state into contiguous one-dimensional Unified Memory buffers and implements the routing/backpropagation operations as explicit kernels. Conceptually structured objects such as grids of vectors and per-edge transformation matrices become addressable slices of flat arrays. CUDA grid/block dimensions are then chosen to represent capsule index, class/parent index, vector dimension, filter coordinate, or matrix coordinate.

That produces a pipeline roughly like:

```text
MNIST image
  -> convolution / primary capsule features
  -> remap scalar feature maps into lower-level capsule vectors
  -> transform each lower-level capsule into votes for candidate parent capsules
  -> iterative routing by agreement
  -> class capsule vectors
  -> margin loss / backpropagation
```

The same repository also contains:

1. a multilayer perceptron built from scratch;
2. a convolutional network built from scratch;
3. a sequential/Armadillo Capsule Network reference;
4. a CUDA Capsule Network using custom kernels and Unified Memory;
5. an NSGA-II implementation for architecture/loss search;
6. PostgreSQL-backed memoization of expensive chromosome evaluations;
7. SLURM/HPC experiment scripts.

## Branch history matters

| Branch | Role |
|---|---|
| `master` | Older sequential/reference state. It does **not** contain the complete thesis CUDA implementation. |
| `CUDAify` | Thesis-era CUDA implementation. Open PR #3 targets `master`; this is the preservation base for current revival work. |
| `smallNorbReader` | Open PR #5 into `CUDAify`; adds a dataset abstraction and smallNORB reader. It remains unmerged and is treated as follow-on work. |
| `hpcvis3-leftover` | A fork recovered from an old UNR machine in 2020. It contains later artifacts but forked before the final `CUDAify` tip, so it is not automatically the canonical successor. |
| `File-IO` | Earlier experimental branch retained for archaeology. |

See [`docs/REPO_AUDIT.md`](docs/REPO_AUDIT.md) before reconciling branches.

## Why revisit Capsule Networks now?

Capsule Networks tried to make the latent ontology of a model more explicit: a capsule vector represents an entity/feature type, vector length is used as an existence/confidence signal, and orientation carries learned instantiation information. Lower capsules produce transformed votes for higher capsules; routing strengthens assignments when votes agree with a candidate parent.

That has a useful family resemblance to attention: vector compatibility, softmax-normalized routing weights, and weighted aggregation decide where information flows. But dynamic routing is iterative, more semantically opinionated, and substantially less hardware-friendly than the dense matrix operations that made Transformers scale so well.

The long-form discussion, including where the analogy holds and where it breaks, is in [`WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md`](WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md).

## Historical performance claims

The thesis reports large CUDA speedups over the sequential reference implementation, including roughly 32x forward-propagation and 116x back-propagation speedup around the original architecture dimensions, with higher peak measurements in the explored range.

Those numbers are historically interesting but must be interpreted with their original methodology. The CPU/reference and CUDA build paths did not use symmetric compiler optimization settings, and the comparison was not a head-to-head against a contemporary highly optimized GPU framework/library implementation. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Important audit status

The revival work does **not** currently claim that the published accuracy/evolution results have been reproduced from the surviving branch tip.

Two code paths deserve special attention before reproducing results:

- the surviving CUDA `tally(false)` path appears to perform backpropagation and periodic weight updates while traversing the test set, despite the thesis describing test evaluation as forward-only;
- the sequential `CapsuleNetwork::backPropagate` accumulation defect is corrected and host-tested in revival PR #7, but CPU/CUDA parity and historical experiment provenance remain unresolved.

These are **audit flags**, not retrospective declarations that the thesis results were generated incorrectly. The exact experiment commit/path must be reconstructed first. See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

## Repository map

```text
include/                         C++/CUDA headers
src/
  CapsuleNetwork/                sequential + CUDA Capsule Network
  ConvolutionalNetwork/          sequential + CUDA convolution
  MultilayerPerceptron/          reference MLP implementation
  GA/                            NSGA-II / chromosome evaluation
  models/                        host structures + CUUnifiedBlob
slurm/                           historical cluster runners
cmds/                            historical experiment helpers
data/                            bundled historical dataset/artifact files
bin/layer_weights/               historical weights

docs/                            human/research documentation
.cursor/                         agent change-making documentation/rules
.agents/ai-skills/               shared ai-skills submodule
SPEC.md                          target restoration state
PLAN.md                          independently attackable work packages
```

## Building

Prefer the Makefile as the developer interface. Contemporary CMake presets cover
host-only and CUDA configure modes. Details, options, and the historical
`sm_30` / `FindCUDA` caveats live in [`docs/BUILD.md`](docs/BUILD.md).

```bash
make ci              # host tests + docs + package check (no GPU required)
make configure-host  # CMake host preset
make cuda-compile    # compile CUDA targets when a toolkit is available
```

Host CI success is **not** GPU runtime validation. See [`docs/CICD.md`](docs/CICD.md)
and [`docs/TESTING.md`](docs/TESTING.md).

See also [`SPEC.md`](SPEC.md), [`PLAN.md`](PLAN.md), and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

### MNIST data location

The reader expects the standard 28x28 MNIST IDX files with these exact names:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

It checks `CAPSNET_DATA_DIR` first, then `data/`, then `../data/`. An explicitly
configured directory must contain all four files; it does not silently fall back.
For example:

```bash
CAPSNET_DATA_DIR=/path/to/mnist ./bin/NeuralNets
```

The loader validates IDX magic numbers, matching image/label counts, dimensions,
payload lengths, and labels in the range 0–9.

## Agent / contributor setup

Clone submodules:

```bash
git clone --recurse-submodules https://github.com/TennisGazelle/CUDA-CapsuleNetwork-Methods.git
```

For agent-assisted work, start with [`AGENTS.md`](AGENTS.md). Shared skills are pinned under `.agents/ai-skills` and wired into `.agents/skills` / `.claude/skills`.

## Restoration principles

1. Preserve the thesis-era artifact before changing behavior.
2. Turn suspected numerical/correctness issues into tests.
3. Establish CPU/GPU primitive parity before kernel optimization.
4. Separate training, evaluation, and benchmarking paths.
5. Reproduce historical measurements with their original caveats.
6. Add corrected/modern benchmarks alongside historical ones, never over them.
7. Only then consider a modern research branch using contemporary CUDA, libraries, attention-inspired routing, or newer capsule formulations.

## Documentation

Start at [`docs/README.md`](docs/README.md).

For the intended end state, read [`SPEC.md`](SPEC.md). For concrete work packages that another agent can pick up independently, read [`PLAN.md`](PLAN.md).
