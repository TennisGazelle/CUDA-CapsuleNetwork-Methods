# Thesis context and contribution map

## Primary artifacts

This repository corresponds to two closely related research artifacts by Daniel Lopez at the University of Nevada, Reno:

1. **A GPU Acceleration Method for Dynamically Routed Capsule Networks**  
   https://www.cse.unr.edu/~fredh/papers/conf/194-amlfdrcl/paper.pdf
2. **Evolving GPU-Accelerated Capsule Networks**  
   https://www.cse.unr.edu/~fredh/papers/thesis/071-lopez/thesis.pdf

The shorter paper focuses on the CUDA representation and acceleration method. The thesis expands the systems work and then uses the accelerated implementation to make multi-objective evolutionary search over Capsule Network parameters computationally practical.

## What the contribution was

The strongest interpretation of the work is not simply:

> "Capsule Networks were reimplemented in CUDA."

The more precise contribution is:

> **A Capsule Network-specific memory representation and CUDA execution model were developed for the vector/matrix operations required by dynamic routing and backpropagation, and the resulting acceleration was used to explore architecture/loss parameters with NSGA-II.**

This matters because the natural object representation of a Capsule Network is hostile to straightforward GPU execution. A sequential implementation wants objects containing vectors, transformation matrices, and routing state. The CUDA implementation instead flattens these structures into contiguous arrays and uses arithmetic indexing plus CUDA grid dimensions to recover their conceptual structure.

The implementation therefore crosses several abstraction layers:

```text
Capsule mathematics
    -> sequential C++/Armadillo reference
    -> explicit flat memory layout
    -> custom CUDA primitives
    -> full forward/backward path
    -> benchmark harness
    -> NSGA-II architecture/loss search
    -> PostgreSQL memoization + SLURM experiment tooling
```

## Why the MLP and CNN are in the repository

The repository was built bottom-up:

- a multilayer perceptron;
- a convolutional network;
- a sequential Capsule Network;
- a CUDA Capsule Network.

That progression gave the author control over data representation, training, and backpropagation instead of relying on a high-level framework whose memory model and operator scheduling would obscure the systems experiment.

The MLP/CNN code should therefore be treated partly as **reference infrastructure and learning lineage**, not as the primary research claim.

## CUDA contribution

The CUDA path introduces a `CUUnifiedBlob` abstraction backed by Unified Memory and implements operations such as:

- matrix-vector multiplication across capsule relationships;
- softmax over coupling/routing logits;
- weighted vote reduction;
- vector squash;
- agreement scalar products;
- margin-loss derivatives;
- transposed matrix-vector error propagation;
- error reductions;
- weight updates;
- convolution forward/backward operations;
- tensor-to-capsule remapping and reconstruction of error tensors.

See [`CUDA_ARCHITECTURE.md`](CUDA_ARCHITECTURE.md).

## Historical performance result

The thesis reports large speedups versus the sequential reference, including roughly 32x forward-propagation and 116x back-propagation speedup near the original capsule dimensions, with peak observations around 33x and 130x in the explored configurations.

The full forward-plus-backward improvement is smaller, roughly on the order of 20x for larger explored configurations.

The interesting systems result is not only the magnitude. Forward and backward propagation scale differently because their computation/communication structure differs. The thesis also observes speedup flattening as tensor-channel count increases, consistent with fixed overhead/serial portions beginning to dominate.

### Interpretation warning

Do not restate those values as "130x faster than state-of-the-art GPU Capsule Networks." The historical benchmark was primarily custom CUDA versus the repository's sequential C++/Armadillo implementation. The surviving CMake configuration also shows asymmetric optimization context: NVCC receives `-O2`, while the historical sequential path was not benchmarked as a maximally optimized native baseline.

This does not erase the evidence that the algorithm had substantial exploitable parallelism. It changes what the comparison proves.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Evolutionary contribution

The thesis uses NSGA-II to explore seven quantities encoded in the chromosome:

- lower/inner capsule dimension;
- upper/outer capsule dimension;
- tensor-channel count;
- batch size;
- margin-loss \(m^+\);
- margin-loss \(m^-\);
- margin-loss \(\lambda\).

The surviving `Individual::decodeChromosome()` maps seven 5-bit regions into those configuration values.

Because neural-network fitness evaluation is expensive and evolutionary algorithms create duplicate individuals, the repository caches evaluated chromosomes in PostgreSQL. That is an important systems detail: duplicate candidates do not need to repeat the entire training/evaluation cost.

A reported evolved configuration used unintuitive dimensions, including a much larger lower capsule and a very small upper capsule, while performing well in the early training criterion. That is exactly the sort of result evolutionary search is useful for discovering.

The evolutionary result should nevertheless be interpreted cautiously. It is stronger as a demonstration that GPU acceleration enables a previously expensive search than as definitive evidence that one evolved topology is universally superior. The thesis itself is cautious about limited observations and convergence.

## Historical branch mapping

The default branch does not tell the complete story.

- `master` contains an older sequential/reference state.
- `CUDAify` is the open PR #3 lineage containing the CUDA thesis implementation and later GA infrastructure.
- `smallNorbReader` is open PR #5 into `CUDAify` and adds a dataset abstraction/smallNORB reader.
- `hpcvis3-leftover` contains a 2020 recovery commit from an old UNR machine but forked from an earlier CUDAify commit; it is an archaeological branch, not automatically the canonical final state.

Revival work starts from `CUDAify` so the thesis implementation remains the baseline.

## What is not yet proven in the revival

The current restoration has not yet established:

- the exact commit used for every thesis figure/table;
- deterministic reproduction of the reported accuracy curves;
- deterministic reproduction of the NSGA-II result set;
- numerical parity between all sequential and CUDA primitives;
- whether the current surviving `tally(false)` mutation behavior was present in the experiment snapshot that produced published results;
- a modern equivalent benchmark with symmetric compiler flags and optimized library baselines.

Those are explicit milestones in [`../PLAN.md`](../PLAN.md), not facts to paper over.
