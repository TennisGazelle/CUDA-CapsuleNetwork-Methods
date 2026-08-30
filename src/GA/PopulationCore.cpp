// Host-testable NSGA-II population semantics.

#include "GA/Population.h"

#include <Utils.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <limits>
#include <stdexcept>

void Population::generate(int n, int bitstringSize) {
    if (n < 0) {
        throw std::invalid_argument("population size cannot be negative");
    }
    clear();
    reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        emplace_back(bitstringSize);
    }
}

void Population::print() {
    if (empty()) {
        cout << "<empty population>" << endl;
        return;
    }

    ParedoFront pointers = ParedoFront::referToAsFront(*this);
    pointers.sortByChromosome();
    for (std::size_t i = 0; i < pointers.size(); ++i) {
        cout << "[" << (i < 10 ? "0" : "") << i << "] - " << pointers[i]->to_string() << endl;
    }
}

void Population::fullPrint() {
    if (empty()) {
        cout << "<empty population>" << endl;
        return;
    }

    ParedoFront pointers = ParedoFront::referToAsFront(*this);
    pointers.sortByChromosome();
    for (std::size_t i = 0; i < pointers.size(); ++i) {
        cout << "[" << (i < 10 ? "0" : "") << i << "] - ";
        pointers[i]->fullPrint();
        cout << endl;
    }
}

void Population::getStatsFromIndividuals() {
    accuracy100.reset();
    accuracy300.reset();
    loss100.reset();
    loss300.reset();

    if (empty()) {
        accuracy100 = PopulationStats{};
        accuracy300 = PopulationStats{};
        loss100 = PopulationStats{};
        loss300 = PopulationStats{};
        return;
    }

    for (const auto& individual : *this) {
        accuracy100.min = std::min(accuracy100.min, individual.accuracy_100);
        accuracy100.max = std::max(accuracy100.max, individual.accuracy_100);
        accuracy300.min = std::min(accuracy300.min, individual.accuracy_300);
        accuracy300.max = std::max(accuracy300.max, individual.accuracy_300);

        loss100.min = std::min(loss100.min, individual.loss_100);
        loss100.max = std::max(loss100.max, individual.loss_100);
        loss300.min = std::min(loss300.min, individual.loss_300);
        loss300.max = std::max(loss300.max, individual.loss_300);

        accuracy100.average += individual.accuracy_100;
        accuracy300.average += individual.accuracy_300;
        loss100.average += individual.loss_100;
        loss300.average += individual.loss_300;
    }

    const double count = static_cast<double>(size());
    accuracy100.average /= count;
    accuracy300.average /= count;
    loss100.average /= count;
    loss300.average /= count;
}

void Population::insertParedoFront(ParedoFront front) {
    for (Individual* individual : front) {
        if (individual != nullptr) {
            push_back(*individual);
        }
    }
}

Individual Population::tournamentSelect(bool useCrowdingOperator) {
    if (size() < 2) {
        throw std::logic_error("tournament selection requires at least two individuals");
    }

    const int leftIndex = Utils::getRandBetween(0, static_cast<int>(size()));
    int rightIndex = Utils::getRandBetween(0, static_cast<int>(size()));
    while (rightIndex == leftIndex) {
        rightIndex = Utils::getRandBetween(0, static_cast<int>(size()));
    }

    const Individual& left = at(static_cast<std::size_t>(leftIndex));
    const Individual& right = at(static_cast<std::size_t>(rightIndex));

    if (useCrowdingOperator) {
        if (left.crowdingOperator(right)) {
            return left;
        }
        if (right.crowdingOperator(left)) {
            return right;
        }
    } else {
        if (left.paredoDominates(right)) {
            return left;
        }
        if (right.paredoDominates(left)) {
            return right;
        }
    }

    return Utils::randomWithProbability(0.5) ? left : right;
}

Individual Population::getBestIndividual() const {
    if (empty()) {
        throw std::logic_error("cannot choose a best individual from an empty population");
    }

    const Individual* best = &front();
    for (const auto& candidate : *this) {
        if (candidate.crowdingOperator(*best)) {
            best = &candidate;
        }
    }
    return *best;
}

unsigned int Population::getNumUniqueIndividuals() const {
    std::vector<std::string> chromosomes;
    chromosomes.reserve(size());
    for (const auto& individual : *this) {
        const std::string chromosome = individual.to_string();
        if (std::find(chromosomes.begin(), chromosomes.end(), chromosome) == chromosomes.end()) {
            chromosomes.push_back(chromosome);
        }
    }
    return static_cast<unsigned int>(chromosomes.size());
}

namespace {

bool hasFiniteObjectives(const Individual& individual) {
    return std::isfinite(individual.accuracy_100) &&
           std::isfinite(individual.accuracy_300) &&
           std::isfinite(individual.loss_100) &&
           std::isfinite(individual.loss_300);
}

template <typename SortFn, typename ValueFn>
void addCrowdingContribution(ParedoFront& front, SortFn sortFn, ValueFn valueFn) {
    sortFn();
    const std::size_t n = front.size();
    if (n == 0) {
        return;
    }

    front.front()->crowdingDistance = std::numeric_limits<double>::infinity();
    front.back()->crowdingDistance = std::numeric_limits<double>::infinity();
    if (n <= 2) {
        return;
    }

    const double minValue = valueFn(*front.front());
    const double maxValue = valueFn(*front.back());
    const double range = std::abs(maxValue - minValue);
    if (range <= std::numeric_limits<double>::epsilon()) {
        return;
    }

    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (std::isinf(front[i]->crowdingDistance)) {
            continue;
        }
        const double previous = valueFn(*front[i - 1]);
        const double next = valueFn(*front[i + 1]);
        front[i]->crowdingDistance += std::abs(next - previous) / range;
    }
}

}  // namespace

void ParedoFront::assignCrowdingDistance() {
    if (empty()) {
        return;
    }

    for (Individual* individual : *this) {
        individual->crowdingDistance = 0.0;
    }

    addCrowdingContribution(*this,
        [this] { sortByAccuracy100(); },
        [](const Individual& i) { return i.accuracy_100; });
    addCrowdingContribution(*this,
        [this] { sortByLoss100(); },
        [](const Individual& i) { return i.loss_100; });
    addCrowdingContribution(*this,
        [this] { sortByAccuracy300(); },
        [](const Individual& i) { return i.accuracy_300; });
    addCrowdingContribution(*this,
        [this] { sortByLoss300(); },
        [](const Individual& i) { return i.loss_300; });

    isSorted = true;
}

void ParedoFront::sortByCrowdingOperator() {
    std::sort(begin(), end(), [](Individual* lhs, Individual* rhs) {
        return lhs->crowdingOperator(*rhs);
    });
}

vector<ParedoFront> sortFastNonDominated(Population &population) {
    for (const Individual& individual : population) {
        if (!hasFiniteObjectives(individual)) {
            throw std::invalid_argument("NSGA-II objectives must be finite");
        }
    }

    vector<ParedoFront> fronts;
    ParedoFront firstFront;

    for (std::size_t i = 0; i < population.size(); ++i) {
        Individual& current = population[i];
        current.individualsIDominate.clear();
        current.numDominateMe = 0;
        current.rank = 0;

        for (std::size_t j = 0; j < population.size(); ++j) {
            if (i == j) {
                continue;
            }
            if (current.paredoDominates(population[j])) {
                current.individualsIDominate.push_back(&population[j]);
            } else if (population[j].paredoDominates(current)) {
                ++current.numDominateMe;
            }
        }

        if (current.numDominateMe == 0) {
            current.rank = 1;
            firstFront.push_back(&current);
        }
    }

    if (firstFront.empty()) {
        return fronts;
    }
    fronts.push_back(firstFront);

    std::size_t frontIndex = 0;
    while (frontIndex < fronts.size()) {
        ParedoFront nextFront;
        for (Individual* dominant : fronts[frontIndex]) {
            for (Individual* dominated : dominant->individualsIDominate) {
                if (dominated->numDominateMe == 0) {
                    continue;
                }
                --dominated->numDominateMe;
                if (dominated->numDominateMe == 0) {
                    dominated->rank = static_cast<unsigned int>(frontIndex + 2);
                    nextFront.push_back(dominated);
                }
            }
        }
        if (nextFront.empty()) {
            break;
        }
        fronts.push_back(nextFront);
        ++frontIndex;
    }

    return fronts;
}

void ParedoFront::sortByAccuracy100() {
    std::sort(begin(), end(), [](Individual* lhs, Individual* rhs) {
        return lhs->accuracy_100 < rhs->accuracy_100;
    });
}

void ParedoFront::sortByAccuracy300() {
    std::sort(begin(), end(), [](Individual* lhs, Individual* rhs) {
        return lhs->accuracy_300 < rhs->accuracy_300;
    });
}

void ParedoFront::sortByLoss100() {
    std::sort(begin(), end(), [](Individual* lhs, Individual* rhs) {
        return lhs->loss_100 < rhs->loss_100;
    });
}

void ParedoFront::sortByLoss300() {
    std::sort(begin(), end(), [](Individual* lhs, Individual* rhs) {
        return lhs->loss_300 < rhs->loss_300;
    });
}

void ParedoFront::sortByChromosome() {
    std::sort(begin(), end(), [](const Individual* lhs, const Individual* rhs) {
        return lhs->to_string() < rhs->to_string();
    });
}

ParedoFront ParedoFront::referToAsFront(Population &population) {
    ParedoFront result;
    result.reserve(population.size());
    for (auto& individual : population) {
        result.push_back(&individual);
    }
    return result;
}

pair<ParedoFront, ParedoFront> ParedoFront::referToUniqueIndividuals(Population &population) {
    ParedoFront unique;
    ParedoFront duplicates;
    unique.reserve(population.size());

    for (auto& individual : population) {
        const bool seen = std::any_of(unique.begin(), unique.end(), [&individual](const Individual* candidate) {
            return candidate->to_string() == individual.to_string();
        });
        (seen ? duplicates : unique).push_back(&individual);
    }
    return {unique, duplicates};
}
