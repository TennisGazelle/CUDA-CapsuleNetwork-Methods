// Host-testable genetic generation and NSGA-II truncation mechanics.

#include "GA/GA.h"

#include <Utils.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

Individual selectParent(Population& parents, bool useCrowdingOperator) {
    if (parents.empty()) {
        throw std::logic_error("cannot select from an empty parent population");
    }
    if (parents.size() == 1) {
        return parents.front();
    }
    return parents.tournamentSelect(useCrowdingOperator);
}

void maybeMutate(Individual& individual, double probability) {
    if (Utils::randomWithProbability(probability)) {
        individual.mutate();
    }
}

}  // namespace

Population makeOffspringGeneration(Population parents,
                                   const GAConfig& config,
                                   bool useCrowdingOperator) {
    if (!std::isfinite(config.prob_mutation) ||
        !std::isfinite(config.prob_crossover) ||
        config.prob_mutation < 0.0 || config.prob_mutation > 1.0 ||
        config.prob_crossover < 0.0 || config.prob_crossover > 1.0) {
        throw std::invalid_argument("GA probabilities must be in [0, 1]");
    }

    Population children;
    children.reserve(parents.size());

    while (children.size() < parents.size()) {
        Individual first = selectParent(parents, useCrowdingOperator);

        if (children.size() + 1 < parents.size() &&
            Utils::randomWithProbability(config.prob_crossover)) {
            Individual second = selectParent(parents, useCrowdingOperator);
            first.crossoverWith(second);
            maybeMutate(first, config.prob_mutation);
            maybeMutate(second, config.prob_mutation);
            children.push_back(first);
            children.push_back(second);
        } else {
            maybeMutate(first, config.prob_mutation);
            children.push_back(first);
        }
    }

    return children;
}

Population selectNextNSGAGeneration(Population combinedPopulation,
                                    std::size_t targetSize) {
    if (targetSize > combinedPopulation.size()) {
        throw std::invalid_argument("target NSGA-II population exceeds candidate pool");
    }

    Population selected;
    selected.reserve(targetSize);
    if (targetSize == 0) {
        return selected;
    }

    std::vector<ParedoFront> fronts = sortFastNonDominated(combinedPopulation);
    for (ParedoFront& front : fronts) {
        if (selected.size() == targetSize) {
            break;
        }

        const std::size_t remaining = targetSize - selected.size();
        if (front.size() <= remaining) {
            selected.insertParedoFront(front);
            continue;
        }

        front.assignCrowdingDistance();
        front.sortByCrowdingOperator();
        for (std::size_t i = 0; i < remaining; ++i) {
            selected.push_back(*front[i]);
        }
        break;
    }

    if (selected.size() != targetSize) {
        throw std::logic_error("NSGA-II truncation did not produce requested population size");
    }
    return selected;
}
