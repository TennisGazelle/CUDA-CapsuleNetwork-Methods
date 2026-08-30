#include <GA/Individual.h>
#include <GA/Population.h>
#include <Utils.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool near(double lhs, double rhs, double tolerance = 1e-10) {
    return std::abs(lhs - rhs) <= tolerance;
}

template <typename Fn>
bool throwsInvalidArgument(Fn fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

Individual makeIndividual(const std::string& bits = std::string(35, '0')) {
    return Individual(35, bits);
}

void setObjectives(Individual& i, double a100, double a300, double l100, double l300) {
    i.accuracy_100 = a100;
    i.accuracy_300 = a300;
    i.loss_100 = l100;
    i.loss_300 = l300;
}

void test_chromosome_decode_boundaries() {
    Individual low = makeIndividual(std::string(35, '0'));
    assert(low.capsNetConfig.cnInnerDim == 2);
    assert(low.capsNetConfig.cnOuterDim == 2);
    assert(low.capsNetConfig.cnNumTensorChannels == 1);
    assert(low.capsNetConfig.batchSize == 20);
    assert(near(low.capsNetConfig.m_plus, 0.8));
    assert(near(low.capsNetConfig.m_minus, 0.00625));
    assert(near(low.capsNetConfig.lambda, 0.4));

    Individual high = makeIndividual(std::string(35, '1'));
    assert(high.capsNetConfig.cnInnerDim == 33);
    assert(high.capsNetConfig.cnOuterDim == 33);
    assert(high.capsNetConfig.cnNumTensorChannels == 32);
    assert(high.capsNetConfig.batchSize == 640);
    assert(near(high.capsNetConfig.m_plus, 0.99375));
    assert(near(high.capsNetConfig.m_minus, 0.2));
    assert(near(high.capsNetConfig.lambda, 0.59375));

    assert(throwsInvalidArgument([] { Individual badSize(34, std::string(34, '0')); }));
    assert(throwsInvalidArgument([] {
        std::string bits(35, '0');
        bits[12] = 'x';
        Individual badChar(35, bits);
    }));
}

void test_seeded_chromosome_generation_is_reproducible() {
    Utils::setRandomSeed(2026);
    Individual first(35);
    Utils::setRandomSeed(2026);
    Individual second(35);
    assert(first.to_string() == second.to_string());
}

void test_pareto_dominance_is_strict() {
    Individual a = makeIndividual();
    Individual b = makeIndividual();
    setObjectives(a, 90.0, 91.0, 1.0, 1.5);
    setObjectives(b, 90.0, 91.0, 1.0, 1.5);

    assert(!a.paredoDominates(b));
    assert(!b.paredoDominates(a));

    setObjectives(a, 91.0, 91.0, 1.0, 1.5);
    assert(a.paredoDominates(b));
    assert(!b.paredoDominates(a));

    setObjectives(b, 95.0, 80.0, 0.5, 2.0);
    assert(!a.paredoDominates(b));
    assert(!b.paredoDominates(a));
}

void test_fast_non_dominated_sort_known_fronts() {
    Population p;
    p.push_back(makeIndividual(std::string(35, '0')));              // A
    p.push_back(makeIndividual(std::string(34, '0') + "1"));       // B
    p.push_back(makeIndividual(std::string(33, '0') + "10"));      // C
    p.push_back(makeIndividual(std::string(33, '0') + "11"));      // D

    setObjectives(p[0], 90.0, 90.0, 1.0, 1.0);  // A dominates B and D.
    setObjectives(p[1], 80.0, 80.0, 2.0, 2.0);  // B dominates D.
    setObjectives(p[2], 95.0, 70.0, 0.5, 3.0);  // C trades off with A.
    setObjectives(p[3], 70.0, 70.0, 4.0, 4.0);  // D.

    const std::vector<ParedoFront> fronts = sortFastNonDominated(p);
    assert(fronts.size() == 3);
    assert(fronts[0].size() == 2);
    assert(fronts[1].size() == 1);
    assert(fronts[2].size() == 1);
    assert(p[0].rank == 1);
    assert(p[2].rank == 1);
    assert(p[1].rank == 2);
    assert(p[3].rank == 3);
}

void test_crowding_distance_prefers_sparse_points() {
    Population p;
    for (int i = 0; i < 5; ++i) {
        std::string bits(35, '0');
        bits[34 - i] = '1';
        p.push_back(makeIndividual(bits));
        setObjectives(p.back(),
                      60.0 + i * 5.0,
                      65.0 + i * 4.0,
                      5.0 - i * 0.5,
                      8.0 - i * 0.75);
        p.back().rank = 1;
    }

    ParedoFront front = ParedoFront::referToAsFront(p);
    front.assignCrowdingDistance();

    int infiniteCount = 0;
    int finiteCount = 0;
    for (Individual* i : front) {
        if (std::isinf(i->crowdingDistance)) {
            ++infiniteCount;
        } else {
            ++finiteCount;
            assert(i->crowdingDistance >= 0.0);
        }
    }
    assert(infiniteCount >= 2);
    assert(finiteCount >= 1);

    front.sortByCrowdingOperator();
    assert(std::isinf(front.front()->crowdingDistance));
}

void test_crowding_operator_prefers_low_rank_then_large_distance() {
    Individual a = makeIndividual();
    Individual b = makeIndividual(std::string(34, '0') + "1");

    a.rank = 1;
    b.rank = 2;
    a.crowdingDistance = 0.0;
    b.crowdingDistance = 100.0;
    assert(a.crowdingOperator(b));
    assert(!b.crowdingOperator(a));

    b.rank = 1;
    a.crowdingDistance = 4.0;
    b.crowdingDistance = 2.0;
    assert(a.crowdingOperator(b));
    assert(!b.crowdingOperator(a));
}

void test_population_stats_and_unique_count() {
    Population p;
    p.push_back(makeIndividual());
    p.push_back(makeIndividual());
    p.push_back(makeIndividual(std::string(34, '0') + "1"));

    setObjectives(p[0], -2.0, 10.0, 3.0, 4.0);
    setObjectives(p[1], 4.0, 20.0, 1.0, 8.0);
    setObjectives(p[2], 1.0, 30.0, 2.0, 6.0);

    p.getStatsFromIndividuals();
    assert(near(p.accuracy100.min, -2.0));
    assert(near(p.accuracy100.max, 4.0));
    assert(near(p.accuracy100.average, 1.0));
    assert(near(p.loss100.min, 1.0));
    assert(near(p.loss100.max, 3.0));
    assert(near(p.loss100.average, 2.0));
    assert(p.getNumUniqueIndividuals() == 2);
}

void test_mutation_changes_exactly_one_bit() {
    Utils::setRandomSeed(17);
    Individual individual = makeIndividual();
    const std::string before = individual.to_string();
    individual.mutate();
    const std::string after = individual.to_string();

    int changed = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
        changed += before[i] != after[i] ? 1 : 0;
    }
    assert(changed == 1);
}

}  // namespace

int main() {
    test_chromosome_decode_boundaries();
    test_seeded_chromosome_generation_is_reproducible();
    test_pareto_dominance_is_strict();
    test_fast_non_dominated_sort_known_fronts();
    test_crowding_distance_prefers_sparse_points();
    test_crowding_operator_prefers_low_rank_then_large_distance();
    test_population_stats_and_unique_count();
    test_mutation_changes_exactly_one_bit();

    std::cout << "host GA/NSGA-II correctness tests passed" << std::endl;
    return 0;
}
