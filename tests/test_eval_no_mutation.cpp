#include <CapsuleNetwork/EvalPolicy.h>

#include <cassert>
#include <iostream>

namespace {

void test_historical_cuda_tally_is_documented_as_mutating() {
    assert(capsnet::historicalCudaTallyMutatesLearnedState());
}

void test_corrected_evaluate_policy_is_pure() {
    assert(!capsnet::correctedEvaluateMutatesLearnedState());
}

void test_default_build_does_not_preserve_historical_eval_mutation() {
    // Corrected default: preserveHistoricalEvalMutation is false unless the
    // CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR CMake option is enabled.
    assert(!capsnet::preserveHistoricalEvalMutation());
}

} // namespace

int main() {
    test_historical_cuda_tally_is_documented_as_mutating();
    test_corrected_evaluate_policy_is_pure();
    test_default_build_does_not_preserve_historical_eval_mutation();
    std::cout << "host evaluation-policy tests passed" << std::endl;
    return 0;
}
