#ifndef NEURALNETS_EVALPOLICY_H
#define NEURALNETS_EVALPOLICY_H

/// Evaluation / train separation policy for Capsule Network passes.
///
/// Historical CUDA `CUCapsuleNetwork::tally(false)` called `backPropagation`
/// and `updateWeights` while traversing the test set. Corrected mode must use
/// a pure `evaluate()` that never mutates learned state.
/// See docs/KNOWN_ISSUES.md and docs/TESTING.md.

namespace capsnet {

/// Returns true: the surviving historical CUDA tally path mutates weights.
bool historicalCudaTallyMutatesLearnedState();

/// Returns false: corrected evaluate must not mutate weights/velocity/deltas.
bool correctedEvaluateMutatesLearnedState();

/// Whether this build prefers historical mutating evaluation semantics.
bool preserveHistoricalEvalMutation();

} // namespace capsnet

#endif
