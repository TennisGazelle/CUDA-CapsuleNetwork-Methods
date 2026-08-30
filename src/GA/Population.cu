// CUDA-dependent population evaluation.

#include "GA/Population.h"

#include <iostream>

namespace {

void copyDuplicateFitness(const ParedoFront& unique, const ParedoFront& duplicates) {
    for (Individual* duplicate : duplicates) {
        for (Individual* original : unique) {
            if (duplicate->to_string() == original->to_string()) {
                *duplicate = *original;
                break;
            }
        }
    }
}

}  // namespace

void Population::evaluate() {
    // The historical implementation advertised a thread pool but selected
    // min(1, unique_count), making it effectively single-threaded while also
    // introducing empty-population/modulo hazards. Keep the corrected default
    // deterministic and explicit. A future GPU-specific parallel evaluator can
    // be added behind parity tests.
    singleThreaded_evaluate();
}

void Population::singleThreaded_evaluate() {
    if (empty()) {
        getStatsFromIndividuals();
        return;
    }

    auto evaluees = ParedoFront::referToUniqueIndividuals(*this);
    cout << "Unique Individuals: " << evaluees.first.size() << endl;

    for (Individual* individual : evaluees.first) {
        individual->evaluate();
    }
    copyDuplicateFitness(evaluees.first, evaluees.second);
    getStatsFromIndividuals();
}

void Population::multiThreaded_evaluate() {
    // Intentionally conservative until simultaneous CUDA-network evaluation is
    // validated on a real GPU. This method remains for API compatibility.
    singleThreaded_evaluate();
}
