#include <GA/GA.h>
#include <Utils.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Fn>
bool throwsInvalidArgument(Fn fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

Individual individualForIndex(int index) {
    std::string bits(35, '0');
    for (int bit = 0; bit < 5; ++bit) {
        if ((index >> bit) & 1) {
            bits[34 - bit] = '1';
        }
    }
    return Individual(35, bits);
}

std::vector<std::string> chromosomes(const Population& p) {
    std::vector<std::string> result;
    result.reserve(p.size());
    for (const auto& individual : p) {
        result.push_back(individual.to_string());
    }
    return result;
}

void setObjectives(Individual& i, double accuracy, double loss) {
    i.accuracy_100 = accuracy;
    i.accuracy_300 = accuracy;
    i.loss_100 = loss;
    i.loss_300 = loss;
}

void test_offspring_generation_does_not_mutate_parents() {
    Population parents;
    for (int i = 0; i < 8; ++i) {
        parents.push_back(individualForIndex(i));
        parents.back().rank = 1;
        parents.back().crowdingDistance = static_cast<double>(i);
        setObjectives(parents.back(), 50.0 + i, 10.0 - i);
    }

    const std::vector<std::string> before = chromosomes(parents);

    GAConfig config;
    config.prob_crossover = 1.0;
    config.prob_mutation = 1.0;

    Utils::setRandomSeed(1234);
    Population children = makeOffspringGeneration(parents, config, true);

    assert(chromosomes(parents) == before);
    assert(children.size() == parents.size());
}

void test_offspring_generation_is_seed_reproducible() {
    Population parents;
    for (int i = 0; i < 6; ++i) {
        parents.push_back(individualForIndex(i));
        setObjectives(parents.back(), 60.0 + i, 6.0 - i * 0.2);
    }

    GAConfig config;
    config.prob_crossover = 0.75;
    config.prob_mutation = 0.4;

    Utils::setRandomSeed(98765);
    const Population first = makeOffspringGeneration(parents, config, false);
    Utils::setRandomSeed(98765);
    const Population second = makeOffspringGeneration(parents, config, false);

    assert(chromosomes(first) == chromosomes(second));
}

void test_offspring_generation_handles_odd_population_exactly() {
    Population parents;
    for (int i = 0; i < 5; ++i) {
        parents.push_back(individualForIndex(i));
        setObjectives(parents.back(), 50.0 + i, 5.0 - i * 0.1);
    }

    GAConfig config;
    config.prob_crossover = 1.0;
    config.prob_mutation = 0.0;

    Utils::setRandomSeed(44);
    const Population children = makeOffspringGeneration(parents, config, false);
    assert(children.size() == 5);
}

void test_nsga_truncation_keeps_complete_better_front() {
    Population candidates;
    for (int i = 0; i < 6; ++i) {
        candidates.push_back(individualForIndex(i));
    }

    // A and B form the first front. C/D trade with each other in front two;
    // E/F are dominated. Selecting three must retain A/B plus one of C/D.
    setObjectives(candidates[0], 95.0, 1.0);
    setObjectives(candidates[1], 90.0, 0.5);
    setObjectives(candidates[2], 80.0, 2.0);
    setObjectives(candidates[3], 85.0, 2.5);
    setObjectives(candidates[4], 70.0, 5.0);
    setObjectives(candidates[5], 60.0, 6.0);

    const std::string a = candidates[0].to_string();
    const std::string b = candidates[1].to_string();
    const Population selected = selectNextNSGAGeneration(candidates, 3);

    assert(selected.size() == 3);
    const std::vector<std::string> selectedBits = chromosomes(selected);
    assert(std::find(selectedBits.begin(), selectedBits.end(), a) != selectedBits.end());
    assert(std::find(selectedBits.begin(), selectedBits.end(), b) != selectedBits.end());
}

void test_nsga_partial_front_keeps_exact_crowding_boundaries() {
    Population candidates;
    for (int i = 0; i < 4; ++i) {
        candidates.push_back(individualForIndex(i));
        // Accuracy and loss both increase, so every candidate is non-dominated.
        // The two endpoint chromosomes are the crowding-distance boundaries.
        setObjectives(candidates.back(), 60.0 + i * 10.0, 1.0 + i);
    }

    const std::string lowBoundary = candidates.front().to_string();
    const std::string highBoundary = candidates.back().to_string();
    const Population selected = selectNextNSGAGeneration(candidates, 2);
    const std::vector<std::string> selectedBits = chromosomes(selected);

    assert(selected.size() == 2);
    assert(std::find(selectedBits.begin(), selectedBits.end(), lowBoundary) != selectedBits.end());
    assert(std::find(selectedBits.begin(), selectedBits.end(), highBoundary) != selectedBits.end());
}

void test_nsga_truncation_boundaries() {
    Population candidates;
    for (int i = 0; i < 3; ++i) {
        candidates.push_back(individualForIndex(i));
        setObjectives(candidates.back(), 70.0 + i, 3.0 - i * 0.1);
    }

    assert(selectNextNSGAGeneration({}, 0).empty());
    assert(selectNextNSGAGeneration(candidates, 0).empty());
    std::vector<std::string> allSelected =
        chromosomes(selectNextNSGAGeneration(candidates, candidates.size()));
    std::vector<std::string> allCandidates = chromosomes(candidates);
    std::sort(allSelected.begin(), allSelected.end());
    std::sort(allCandidates.begin(), allCandidates.end());
    assert(allSelected == allCandidates);

    bool threw = false;
    try {
        selectNextNSGAGeneration(candidates, candidates.size() + 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_nsga_truncation_does_not_mutate_input() {
    Population candidates;
    for (int i = 0; i < 4; ++i) {
        candidates.push_back(individualForIndex(i));
        setObjectives(candidates.back(), 80.0 + i, 4.0 - i * 0.25);
    }
    const std::vector<std::string> before = chromosomes(candidates);

    const Population selected = selectNextNSGAGeneration(candidates, 2);
    assert(selected.size() == 2);
    assert(chromosomes(candidates) == before);
}

void test_invalid_probabilities_are_rejected() {
    Population parents;
    parents.push_back(individualForIndex(0));

    GAConfig config;
    config.prob_mutation = 1.1;
    bool threw = false;
    try {
        makeOffspringGeneration(parents, config, false);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    config.prob_mutation = 0.0;
    config.prob_crossover = std::numeric_limits<double>::quiet_NaN();
    assert(throwsInvalidArgument([&parents, &config] {
        makeOffspringGeneration(parents, config, false);
    }));

    config.prob_crossover = std::numeric_limits<double>::infinity();
    assert(throwsInvalidArgument([&parents, &config] {
        makeOffspringGeneration(parents, config, false);
    }));
}

void test_probability_boundaries_are_accepted() {
    Population parents;
    parents.push_back(individualForIndex(0));

    GAConfig config;
    config.prob_crossover = 0.0;
    config.prob_mutation = 0.0;
    assert(makeOffspringGeneration(parents, config, false).size() == 1);

    config.prob_crossover = 1.0;
    config.prob_mutation = 1.0;
    assert(makeOffspringGeneration(parents, config, false).size() == 1);
}

}  // namespace

int main() {
    test_offspring_generation_does_not_mutate_parents();
    test_offspring_generation_is_seed_reproducible();
    test_offspring_generation_handles_odd_population_exactly();
    test_nsga_truncation_keeps_complete_better_front();
    test_nsga_partial_front_keeps_exact_crowding_boundaries();
    test_nsga_truncation_boundaries();
    test_nsga_truncation_does_not_mutate_input();
    test_invalid_probabilities_are_rejected();
    test_probability_boundaries_are_accepted();

    std::cout << "host GA generation tests passed" << std::endl;
    return 0;
}
