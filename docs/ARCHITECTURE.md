# Repository architecture

This document describes the surviving `CUDAify` code as a historical implementation. It is a map for restoration work, not a claim that every path is correct or active in the thesis experiments.

## 1. Architectural layers

The repository was developed bottom-up:

```text
MultilayerPerceptron
        |
        v
ConvolutionalNetwork
        |
        v
sequential CapsuleNetwork (Armadillo)
        |
        v
CUDA CapsuleNetwork (CUUnifiedBlob + custom kernels)
        |
        v
NSGA-II architecture/loss search
```

That lineage matters. The sequential code is both a usable implementation and the closest surviving numerical oracle for the CUDA path.

## 2. Configuration

`include/CapsNetConfig.h` centralizes the main Capsule Network parameters:

- input: 28x28;
- classes: 10;
- routing iterations: 3;
- default lower/inner capsule dimension: 8;
- default upper/outer capsule dimension: 16;
- tensor-channel count: 2 in the surviving tip;
- batch size: 250;
- margin loss: `m_plus=0.9`, `m_minus=0.1`, `lambda=0.5`;
- learning rate: 0.1.

Several values that look configurable are still coupled to hard-coded `28`, `6x6`, or 10-class assumptions elsewhere. Do not infer generality solely from `CapsNetConfig`.

## 3. Sequential Capsule Network

Primary files:

- `include/CapsuleNetwork/CapsuleNetwork.h`
- `src/CapsuleNetwork/CapsuleNetwork.cpp`
- `include/CapsuleNetwork/Capsule.h`
- `src/CapsuleNetwork/Capsule.cpp`
- `models/VectorMap.*`

The sequential path uses Armadillo vectors/matrices for capsule state.

### Forward path

At a high level:

1. load an MNIST image;
2. pass it through the convolutional `primaryCaps` layer;
3. reinterpret groups of feature maps as lower-level capsule vectors;
4. squash those vectors;
5. for each digit/class capsule, transform all lower capsule vectors into votes;
6. apply dynamic routing;
7. return one output vector per class.

Classification uses the class capsule with the largest vector length.

### Routing object

Each sequential `Capsule` owns:

- transformation matrices for its incoming lower capsules;
- routing logits `b`;
- coupling coefficients `c`;
- weight deltas and momentum/velocity state;
- previous input and output.

`Capsule::routingAlgorithm()` follows the familiar sequence:

```text
u -> u_hat
initialize b
repeat routing iterations:
  c = softmax(b)
  v = squash(sum(c * u_hat))
  b += dot(u_hat, v)
```

This object-oriented representation is readable but creates many separately managed vectors and matrices, which motivated the CUDA redesign.

## 4. CUDA Capsule Network

Primary files:

- `include/CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h`
- `src/CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.cu`
- `include/models/CUUnifiedBlob.h`
- `src/models/CUUnifiedBlob.cu`
- `include/ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.h`
- `src/ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.cu`

Rather than allocating one C++ object per capsule relationship, the CUDA path stores logically structured state in flat `CUUnifiedBlob` buffers.

`CUCapsuleNetwork` owns blobs for:

- `u`: lower-level capsule state;
- `u_hat`: transformed votes;
- `w`: transformation matrices;
- `w_delta`: accumulated weight errors;
- `w_velocity`: momentum state;
- `v`: class/upper capsule output (and later error state in portions of backprop);
- `b`: routing logits;
- `c`: routing coefficients;
- `truth`: class target vector;
- `losses`: per-class loss scratch;
- `lengths`: per-class vector lengths;
- `cache`: debugging/scratch state.

See [`CUDA_ARCHITECTURE.md`](CUDA_ARCHITECTURE.md) for dimensions and kernel mapping.

## 5. Convolutional path

The repository contains a hand-written convolutional implementation as well as a CUDA convolutional layer used to produce primary capsule features.

The thesis explicitly identifies this convolution code as a comparatively naive component and a future optimization target. It should not be mistaken for a cuDNN-equivalent implementation.

The output is reshaped/remapped so sets of scalar channels become vectors. This is where a conventional convolutional tensor becomes the lower-level capsule tensor.

## 6. Loss and training

The class capsules use margin loss based on vector length. Weight updates use a momentum-like velocity/delta state rather than delegating optimization to a framework.

The repository includes a reconstruction-network implementation in the sequential path, but reconstruction-related training is partially commented out in the surviving branch and therefore must be reconstructed historically before being treated as part of every experiment.

### Training/evaluation caution

The surviving CUDA `tally()` implementation mutates state by invoking backpropagation and weight updates. That conflicts with the thesis description of forward-only test evaluation and is a critical audit target. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## 7. NSGA-II layer

Primary files:

- `include/GA/Individual.h`, `src/GA/Individual.cu`
- `include/GA/Population.h`, `src/GA/Population.cu`
- `include/GA/GA.h`, `src/GA/GA.cu`
- `include/GA/CapsNetDAO.h`, `src/GA/CapsNetDAO.cpp`
- `include/GAConfig.h`

Each `Individual` is a binary chromosome. Seven 5-bit segments are decoded into Capsule Network configuration values. The exact mapping should be read from `Individual::decodeChromosome()` when reproducing the thesis because comments, bounds, and historical defaults changed during development.

The NSGA-II implementation includes:

- tournament selection;
- mutation;
- crossover;
- Pareto dominance;
- non-dominated sorting;
- crowding distance;
- parent/child population construction.

Fitness includes accuracy/loss observations at two evaluation horizons in the surviving model.

### Result memoization

`CapsNetDAO` stores evaluated chromosomes in PostgreSQL. This is a practical optimization because evolutionary search can generate duplicate chromosomes and neural-network fitness evaluation is expensive.

The database is therefore part of the historical experiment infrastructure, not part of the fundamental Capsule Network inference algorithm.

## 8. Experiment entry points

`src/main.cu` is an accumulation of experiment functions rather than a modern CLI. Historically it was used to switch among:

- sequential/CUDA verification;
- detailed forward propagation tests;
- timing/speedup experiments;
- training/tally experiments;
- GA runs;
- threading/stream experiments.

Do not assume the currently uncommented call is the call used for a published figure. Reproduction work should trace commit history and experiment-specific functions.

## 9. Cluster tooling

`slurm/` and `cmds/` contain historical helpers for running on UNR HPC systems. They include environment/path assumptions that may no longer exist.

They should be preserved as evidence and eventually wrapped/documented rather than rewritten in-place until the original environment is understood.

## 10. Data and artifacts

The repository contains historical MNIST data files and weight artifacts directly in Git. This is convenient for archaeology but not a modern dependency-management pattern.

A restoration should distinguish:

- source code required to build;
- small deterministic test fixtures;
- externally downloadable datasets;
- generated training weights;
- published benchmark result data.

Do not delete historical artifacts until a replacement provenance/download story is established.

## 11. Architectural seams useful for modernization

The best seams for incremental work are:

1. **pure math utilities** (`Utils`);
2. **individual CUDA primitives** (`CUUnifiedBlob` static operations);
3. **routing iteration**;
4. **full capsule forward pass**;
5. **full capsule backward pass**;
6. **dataset/evaluation boundary**;
7. **benchmark harness**;
8. **GA fitness evaluation**.

The plan intentionally moves in that order so correctness can be established below the level of full training curves.
