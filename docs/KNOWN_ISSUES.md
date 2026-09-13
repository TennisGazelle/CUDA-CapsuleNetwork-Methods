# Known issues and audit flags

This file deliberately distinguishes **confirmed implementation facts**, **strongly suspected defects**, and **historical questions not yet resolved**. The restoration should convert these items into tests before changing numerical behavior.

## Severity legend

- **Critical:** could alter evaluation validity or published-result interpretation.
- **High:** likely numerical/correctness bug or major reproducibility obstacle.
- **Medium:** behavior is surprising/fragile but may be intentional or outside the published path.
- **Low:** maintainability/performance issue with limited correctness impact.

---

## Critical: test/evaluation path appears to mutate model weights

**Files:** `src/CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.cu`

**Status (corrected path):** PR modernization adds `CUCapsuleNetwork::evaluate()`,
which performs forward + metrics only. Corrected `train()` calls `evaluate(false)`
unless `CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR` is enabled. Host policy coverage is
in `tests/test_eval_no_mutation.cpp`. **GPU weight-snapshot parity that
`evaluate` leaves `w` / velocity / conv filters bit-identical is still required
on a CUDA host** before claiming runtime proof.

The thesis describes evaluating the testing set using forward propagation to obtain an unbiased estimate. In the surviving `CUDAify` tip, historical `CUCapsuleNetwork::tally(bool useTraining)` appears to:

1. forward propagate an example;
2. accumulate loss and correctness;
3. call `backPropagation(imageIndex, useTraining)`;
4. periodically call `updateWeights()`.

When `useTraining == false`, this means the testing labels appear to participate in backpropagation/weight updates while the test set is being traversed.

`CUCapsuleNetwork::train()` also needs exact path reconstruction because surviving code has `runEpoch()` commented in a relevant historical section and historically used `tally(false)` for history/evaluation.

### Why this is an audit flag rather than a thesis verdict

The repository was actively changing around the experiment period. We have not yet proven which commit/executable path generated each published figure. The correct next step is:

- reconstruct the experiment commit(s);
- add a GPU test asserting that pure `evaluate` leaves weights bit-identical;
- keep historical `tally` available under an explicit compatibility switch;
- document whether reproduced historical figures require the mutating behavior.

Do **not** silently remove `tally` and then claim the historical results reproduced.

---

## High: suspicious sequential backpropagation accumulation index

**Status:** Corrected in PR #7 and covered by `test_backprop`. The corrected
path accumulates each upper-capsule contribution by primary-capsule index and
validates every incoming error-vector shape before mutating gradients.

**File:** `src/CapsuleNetwork/CapsuleNetwork.cpp`

In the surviving sequential backpropagation path, a nested loop iterates over `j` but appears to accumulate using `primaryCapsError[i] += subset[i]` rather than indexing by the inner loop variable.

That shape is suspicious because the operation is expected to collect contributions for each lower capsule across upper capsules/classes.

### Required verification

- construct a tiny network with dimensions small enough to calculate by hand;
- compare the sequential result against an independently implemented reference;
- compare against CUDA reduction semantics;
- inspect earlier commits to determine whether this was present in published benchmarks.

The host reduction semantics are now verified. CUDA parity and historical
experiment provenance remain open, so this correction can change sequential
training results and is not evidence that published results have been reproduced.

---

## High: random-weight helper does not appear to use its computed scale

**Status:** Corrected in PR #7 and covered by
`test_weight_initialization_uses_requested_scale`. Corrected mode samples from
a zero-mean normal distribution with standard deviation `0.8 / n` and rejects
non-finite or non-positive scale inputs. Historical runs may have used the
unscaled standard-normal behavior.

**File:** `src/Utils.cpp`

`Utils::getWeightRand(double n)` computes a distribution half-width and standard deviation based on `n`, but the returned value comes from a static standard normal distribution. The computed scale is not applied to the returned sample.

Additionally, callers historically pass values such as `0` in some CUDA initialization paths, making the intermediate `2.4/n` expression problematic even though the computed value is not subsequently used to scale the sample.

This can materially affect initialization and therefore training/reproducibility.

---

## High: static real-valued RNG distribution captures first-call bounds

**Status:** Corrected in PR #7 and covered by
`test_rng_is_seedable_and_bounds_are_per_call`. Each call now constructs a
distribution for its requested bounds. This changes any historical path that
relied on varying real-valued ranges after the first call.

**File:** `src/Utils.cpp`

`Utils::getRandBetween(double lowerBound, double upperBound)` declares a `static uniform_real_distribution` constructed from the function arguments. In C++, that distribution is initialized only on the first call, so subsequent calls with different bounds reuse the first call's distribution.

If this helper is used with varying ranges, later calls do not honor their requested bounds.

---

## Medium/High: vector length includes epsilon twice

**Status:** Corrected in PR #7 and covered by `test_corrected_norm_semantics`.
`square_length` and `length` now implement the literal squared norm and norm;
zero-safe normalization handles the division guard explicitly. Historical
training may differ near zero and requires compatibility treatment if needed.

**File:** `src/Utils.cpp`

`Utils::square_length()` returns the sum of squares plus `EPSILON`. `Utils::length()` then computes `sqrt(square_length(v) + EPSILON)`, adding another epsilon.

The resulting numerical behavior differs from the literal mathematical norm and from implementations that add one epsilon only for safe division.

This may be intentional numerical protection, historical drift, or an accidental double offset. Record current values in tests before modifying.

---

## Medium/High: `asCapsuleVectors` assertion does not protect the later indexing pattern

**Status:** Corrected in PR #7 and covered by
`test_flatten_and_capsule_round_trip`. Corrected mode requires an exact shape
match and rejects invalid dimensions instead of relying on a debug assertion.

**File:** `src/Utils.cpp`

The function asserts:

```cpp
assert(data.size() <= dim * numVectors);
```

but then unconditionally reads `dim * numVectors` elements. A too-small input satisfies the assertion and can then be indexed out of bounds.

The intended condition appears closer to equality (or at minimum `>=` for a deliberate prefix conversion).

---

## Medium: `CUUnifiedBlob` managed-memory ownership needs a leak/lifetime audit

**Files:** `include/models/CUUnifiedBlob.h`, `src/models/CUUnifiedBlob.cu`

`allocateMemory()` allocates both the main `data` buffer and a managed `flagHelper`. The historical deallocation path visibly frees `data`; the `flagHelper` lifetime should be checked carefully for a corresponding free on every allocation/resize/destruction path.

The class also implements copy/assignment/resize semantics manually. Add sanitizer/ownership tests once a CUDA runtime test environment is available.

---

## Medium: evaluation and training responsibilities are mixed

The historical API exposes operations such as `tally`, `train`, `runEpoch`, `forwardAndBackPropagation`, and test/debug methods without a strict modern boundary between:

- pure inference;
- metric calculation;
- backpropagation;
- weight mutation;
- benchmark timing.

This makes accidental state mutation easy and complicates reproducibility. The modernization spec requires explicit separation.

---

## Medium: many dimensions remain hard-coded

Examples include:

- 28x28 input assumptions;
- expressions such as `28-6`;
- `6*6` flattened spatial assumptions;
- ten-class assumptions;
- historical thread arrays sized around class count;
- remapping dimensions embedded in test/experiment code.

`CapsNetConfig` gives the appearance of a general configuration object, but the full codebase is not yet dimension-generic.

---

## Medium: reconstruction network activation is historically ambiguous

The sequential code contains reconstruction-network machinery resembling the original CapsNet decoder regularizer. Portions of training/backprop related to reconstruction are commented out in the surviving branch.

Do not state that every thesis experiment used reconstruction loss until the experiment commit/path confirms it.

---

## Medium: GA contains fake/test evaluation paths alongside real evaluation

**Files:** `src/GA/Individual.cu`, `src/GA/*.cu`

`Individual` contains both real network construction/training and a `fakeEvaluate()` path used during development. The active path and commit used for thesis experiments must be traced rather than inferred from method names.

---

## Medium: NSGA-II implementation needs independent correctness tests

**Status:** Core host semantics are corrected and covered in PR #7 by
`test_ga` and `test_ga_generation`, including known fronts, crowding distance,
exact partial-front truncation, crossover, probability boundaries, non-finite
objective rejection, odd population sizes, reproducibility, and parent
immutability. Database-backed evaluation and historical search provenance
remain separate audit work.

The repository implements non-dominated sorting, crowding distance, tournament selection, crossover, mutation, and front filling manually.

Potential audit areas include:

- Pareto dominance semantics;
- crowding sort direction;
- rank/crowding comparison direction;
- parent/child population truncation;
- crossover modifying a selected parent object;
- random-index upper bounds;
- duplicate handling/database reuse.

The restoration should verify NSGA-II with small synthetic populations whose fronts are known exactly.

---

## Medium: historical performance baseline is not symmetric

The thesis performance comparison demonstrates substantial parallelization, but the surviving build context includes `-O2` for NVCC while the sequential reference was not presented as a maximally optimized C++ baseline. It also was not compared primarily against an optimized GPU framework implementation.

Do not relabel historical speedups as state-of-the-art GPU-vs-GPU speedups.

---

## Medium: obsolete CUDA architecture/toolchain assumptions

`CMakeLists.txt` uses:

- old `FindCUDA` / `cuda_add_executable` style;
- `cmake_minimum_required(VERSION 2.6)`;
- `-arch sm_30`;
- old compiler workaround macros.

Modern NVCC versions have dropped support for `sm_30`. This is a build modernization issue, not an algorithm defect.

---

## Low/Performance: pervasive `cudaDeviceSynchronize()`

Synchronization appears throughout helper/wrapper methods. This simplifies debugging but prevents overlap and can inflate launch overhead.

Before removal, classify each synchronization as a true dependency, a host-read requirement, or historical debugging conservatism.

---

## Low/Performance: helper kernels perform disproportionate launch work

Examples include operations that launch one block per element to clear a buffer or a large launch to set a single element. These were reasonable research scaffolding but are not production GPU primitives.

Preserve them until parity tests exist; then optimize behind the same tests.

---

## Historical branch gaps

### `smallNorbReader`

Open PR #5 adds a `DataReader` abstraction and smallNORB support but is not merged into `CUDAify`. It also contains code that deserves its own correctness audit (e.g. ownership/index bounds) before reconciliation.

### `hpcvis3-leftover`

A 2020 recovery commit contains a large set of artifacts and an experiment with `--default-stream per-thread`, but the branch forked from an earlier `CUDAify` commit. It should be mined selectively, not fast-forwarded as "the newest version."

---

## README-era historical warning

The old `CUDAify` README explicitly said unit tests were pending and told the reader to ignore a destructor segmentation fault because it did not affect assignment speedup measurements.

That is valuable historical context: the code should be treated as research software whose correctness needs to be re-established at the primitive level.
