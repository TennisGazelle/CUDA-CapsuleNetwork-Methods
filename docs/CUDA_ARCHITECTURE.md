# CUDA architecture and memory model

The CUDA implementation's central idea is to replace a comfortable object graph of capsule vectors/matrices with flat, contiguous buffers whose logical dimensions are recovered by arithmetic indexing.

This is the most distinctive systems contribution in the repository.

## 1. `CUUnifiedBlob`

`CUUnifiedBlob` is a manually managed one-dimensional array of `double` allocated with `cudaMallocManaged`.

The same physical representation can be interpreted as:

- a scalar array;
- a set of vectors;
- a grid of vectors;
- a collection of matrices;
- a tensor flattened across spatial/capsule/class dimensions.

Instead of introducing a separate GPU container type for every conceptual shape, the code passes shape information as kernel parameters and uses index arithmetic.

This resembles a primitive tensor abstraction implemented specifically for the Capsule Network workload.

## 2. Main logical buffers

Let:

- `I = cnInnerDim`;
- `O = cnOuterDim`;
- `K = numClasses`;
- `T = flattenedTensorSize = 6 * 6 * cnNumTensorChannels` in the historical MNIST architecture.

The main buffers are sized approximately as:

| Buffer | Concept | Element count |
|---|---|---:|
| `u` | lower capsule vectors repeated/addressed for class relationships | `I * K * T` |
| `w` | transformation matrices | `I * O * K * T` |
| `w_delta` | accumulated transformation-matrix update/error | `I * O * K * T` |
| `w_velocity` | momentum state | `I * O * K * T` |
| `u_hat` | predicted upper capsule vectors/votes | `O * K * T` |
| `v` | final upper/class capsule vectors | `K * O` |
| `b` | routing logits | `K * T` |
| `c` | routing coefficients | `K * T` |
| `truth` | target class indicator | `K` |
| `losses` | per-class loss scratch | `K` |
| `lengths` | per-class vector length scratch | `K` |

The implementation uses `double`, so raw memory pressure grows quickly with capsule count and dimensions.

## 3. Forward propagation mapping

### 3.1 Convolution to lower capsules

The CUDA convolutional layer produces scalar feature maps. `squashAndRemapToU` groups/remaps those scalar values into lower-level capsule vectors and applies the vector squash.

The historical layout repeats/organizes lower capsule data to make class-conditioned transformation operations convenient.

### 3.2 Vote generation

`CUDA_matrixVectorMultiplication(w, u, u_hat, I, O, K, T)` applies the learned transformation matrices.

The associated kernel maps logical tensor/capsule and class indices onto CUDA grid dimensions, with an output-vector coordinate handled by a thread dimension. This turns the dense set of small matrix-vector products into many independent GPU work items.

Conceptually:

\[
\hat{u}_{j|i} = W_{ij}u_i.
\]

### 3.3 Routing softmax

`CUDA_vectorVectorSoftmax(b, c, K, T)` normalizes routing logits into coupling coefficients.

The code uses shared-memory/reduction-style logic to obtain sums across the relevant routing dimension.

### 3.4 Weighted vote reduction

`CUDA_weightReduceVectors(u_hat, c, v, K, T, O)` combines lower votes into an upper capsule candidate:

\[
s_j = \sum_i c_{ij}\hat{u}_{j|i}.
\]

Reduction is one of the important synchronization-sensitive operations in routing.

### 3.5 Squash

`CUDA_vectorSquash(v, K, O)` computes the norm-dependent capsule activation for each class vector.

This requires a reduction across vector coordinates followed by a scaling of every coordinate.

### 3.6 Agreement

`CUDA_vectorVectorScalarProduct(u_hat, v, b, K, T, O)` computes vote-parent agreement and updates routing logits:

\[
b_{ij} \mathrel{+}= \hat{u}_{j|i}\cdot v_j.
\]

The softmax/reduce/squash/agreement cycle repeats `numIterations` times (three in the historical default).

## 4. Backpropagation mapping

The CUDA backward path implements the derivative chain explicitly rather than relying on automatic differentiation.

Important operations include:

- `CUDA_vectorLossFunction`: transforms the class capsule output into an error/gradient-like state using margin-loss and squash-related factors;
- `CUDA_vectorVectorMatrixProductAndSum`: accumulates transformation-matrix errors from output error and previous lower capsule state;
- `CUDA_scaledDecompositionOfError`: distributes upper error according to coupling coefficients;
- `CUDA_weightedTransMatrixVecMult`: applies transposed transformation matrices to propagate error toward lower capsules;
- `CUDA_multiVectorReduction`: reduces class-conditioned lower errors back toward a single lower-capsule error representation;
- `CUDA_vectorSquashDerivative`: applies squash derivative scaling;
- convolutional remapping/backprop kernels: reconstruct the convolutional error tensor and update convolution filters.

The implementation frequently reuses buffers for different semantic states. That reduces allocation churn but makes correctness reasoning harder. Tests should therefore assert values at the boundary of every operation before refactoring buffer lifetimes.

## 5. Weight updates

`w`, `w_delta`, and `w_velocity` implement a momentum-like update. A GPU kernel updates every element independently and clears the accumulated error/delta afterward.

The sequential path maintains analogous matrix-level state.

A future parity harness should start from identical weights and a single deterministic example, then compare:

1. forward `u_hat`;
2. each routing iteration's `c`/`v`/`b`;
3. output loss;
4. backward lower-capsule error;
5. `w_delta`;
6. post-update `w`.

## 6. Why Unified Memory was attractive

Unified Memory made a research implementation simpler because host-side debugging/printing and GPU-side kernels could address the same allocation. It reduced explicit host/device copy bookkeeping while the data model was still evolving.

That convenience has costs:

- page migration behavior can affect performance;
- host access can trigger synchronization/migration;
- explicit placement/prefetch behavior is absent in the historical code;
- frequent `cudaDeviceSynchronize()` calls limit overlap;
- debug helpers can accidentally dominate timings if included incorrectly.

A modern performance pass should measure page-fault/migration behavior before assuming managed memory is still the right representation.

## 7. Synchronization pattern

The code deliberately calls `cudaDeviceSynchronize()` in many wrapper/helper paths. This simplified correctness/debugging, but it serializes stages and prevents the runtime from overlapping independent work.

The thesis already identified concurrency/streaming as future work, and branch history includes experiments around per-thread default streams and CPU threading.

Modernization should not simply delete synchronization. First classify each barrier as:

- required data dependency;
- host-read dependency;
- debugging safety barrier;
- unnecessary historical conservatism.

Then remove/fuse only barriers proven unnecessary.

## 8. Kernel inventory

The surviving `CUUnifiedBlob.cu` includes kernels for or related to:

- clearing buffers;
- NaN detection;
- single-element setting;
- matrix-vector multiplication;
- routing softmax;
- weighted vote reduction;
- capsule squash;
- vote-parent scalar products;
- margin-loss/error transformation;
- transposed matrix-vector multiplication;
- decomposing/scaling error by routing coefficients;
- transformation-matrix gradient accumulation;
- multi-vector reduction;
- momentum/weight updates;
- squash derivative;
- convolution dot products;
- tensor flatten/remap to capsule vectors;
- reconstruction/remapping of error tensors;
- convolution backpropagation;
- vector length computation;
- per-vector loss computation.

This inventory should eventually become executable parity coverage rather than documentation only.

## 9. Historical inefficiencies that should remain visible

Several helpers launch far more work or synchronize more often than a modern implementation would. Examples include clearing a buffer with one block per element and setting one element using a launch sized to the entire buffer.

These are modernization opportunities, not reasons to erase the reference implementation. A useful restoration should retain a `historical` or reference path so speedups from modern kernel engineering can be measured honestly.

## 10. Modern GPU opportunities after parity exists

Only after numerical parity and benchmark methodology are established should a second-phase project consider:

- contemporary CMake CUDA language support;
- modern compute architectures instead of `sm_30`;
- explicit memory placement/prefetch or device allocations;
- operation fusion;
- fewer synchronization barriers;
- cuBLAS for suitable batched matrix operations;
- cuDNN for convolution;
- CUDA Graphs for repeated fixed execution;
- mixed precision where numerically valid;
- Triton/CUTLASS-style alternatives for fused routing primitives;
- sparse/local routing to reduce the algorithmic relationship graph.

Those are research/optimization changes, not repository-hygiene changes.
