#include <CapsuleNetwork/EvalPolicy.h>

namespace capsnet {

bool historicalCudaTallyMutatesLearnedState() {
    return true;
}

bool correctedEvaluateMutatesLearnedState() {
    return false;
}

bool preserveHistoricalEvalMutation() {
#if defined(CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR) && CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR
    return true;
#else
    return false;
#endif
}

} // namespace capsnet
