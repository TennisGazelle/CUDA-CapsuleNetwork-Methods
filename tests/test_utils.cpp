#include <Utils.h>

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

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

void test_reverse_int() {
    const int input = 0x01020304;
    const int expected = 0x04030201;
    assert(Utils::reverseInt(input) == expected);
}

void test_corrected_norm_semantics() {
    arma::vec v(2);
    v[0] = 3.0;
    v[1] = 4.0;

    assert(near(Utils::square_length(v), 25.0));
    assert(near(Utils::length(v), 5.0));

    arma::vec zero(3, arma::fill::zeros);
    assert(near(Utils::square_length(zero), 0.0));
    assert(near(Utils::length(zero), 0.0));
    assert(arma::approx_equal(Utils::safeNormalise(zero), zero, "absdiff", 0.0));
    assert(arma::approx_equal(Utils::squish(zero), zero, "absdiff", 0.0));
}

void test_squish_is_finite_direction_preserving_and_bounded() {
    arma::vec v(2);
    v[0] = 3.0;
    v[1] = 4.0;

    const arma::vec squashed = Utils::squish(v);
    assert(squashed.n_elem == 2);
    assert(std::isfinite(squashed[0]));
    assert(std::isfinite(squashed[1]));
    assert(squashed[0] > 0.0);
    assert(squashed[1] > 0.0);
    assert(near(squashed[0] / squashed[1], 3.0 / 4.0));
    assert(Utils::length(squashed) < 1.0);
}

void test_flatten_and_capsule_round_trip() {
    std::vector<arma::vec> capsules(2, arma::vec(3));
    capsules[0][0] = 1.0;
    capsules[0][1] = 2.0;
    capsules[0][2] = 3.0;
    capsules[1][0] = 4.0;
    capsules[1][1] = 5.0;
    capsules[1][2] = 6.0;

    const std::vector<double> flat = Utils::getAsOneDim(capsules);
    assert(flat.size() == 6);

    const std::vector<arma::vec> rebuilt = Utils::asCapsuleVectors(3, 2, flat);
    assert(rebuilt.size() == capsules.size());
    for (std::size_t i = 0; i < capsules.size(); ++i) {
        for (std::size_t j = 0; j < capsules[i].n_elem; ++j) {
            assert(near(rebuilt[i][j], capsules[i][j]));
        }
    }

    const std::vector<arma::vec> empty;
    assert(Utils::getAsOneDim(empty).empty());
    assert(Utils::asCapsuleVectors(3, 0, {}).empty());

    assert(throwsInvalidArgument([] {
        Utils::asCapsuleVectors(3, 2, {1.0, 2.0, 3.0});
    }));
    assert(throwsInvalidArgument([] {
        Utils::asCapsuleVectors(0, 2, {});
    }));
}

void test_binary_decode() {
    assert(Utils::getBinaryAsInt({}) == 0);
    assert(Utils::getBinaryAsInt({false, false, false, false, false}) == 0);
    assert(Utils::getBinaryAsInt({true, false, false, false, false}) == 16);
    assert(Utils::getBinaryAsInt({true, true, true, true, true}) == 31);
    assert(Utils::getBinaryAsInt({true, false, true, false, true}) == 21);
    assert(throwsInvalidArgument([] {
        Utils::getBinaryAsInt(std::vector<bool>(32, true));
    }));
}

void test_rng_is_seedable_and_bounds_are_per_call() {
    Utils::setRandomSeed(12345);
    const int intA = Utils::getRandBetween(2, 7);
    const double realA = Utils::getRandBetween(-2.0, -1.0);
    const bool boolA = Utils::randomWithProbability(0.25);

    Utils::setRandomSeed(12345);
    assert(Utils::getRandBetween(2, 7) == intA);
    assert(near(Utils::getRandBetween(-2.0, -1.0), realA));
    assert(Utils::randomWithProbability(0.25) == boolA);

    Utils::setRandomSeed(7);
    for (int i = 0; i < 100; ++i) {
        const int value = Utils::getRandBetween(2, 7);
        assert(value >= 2);
        assert(value < 7);
    }

    for (int i = 0; i < 50; ++i) {
        const double firstRange = Utils::getRandBetween(-10.0, -9.0);
        const double secondRange = Utils::getRandBetween(100.0, 101.0);
        assert(firstRange >= -10.0 && firstRange < -9.0);
        assert(secondRange >= 100.0 && secondRange < 101.0);
    }

    assert(throwsInvalidArgument([] { Utils::getRandBetween(2, 2); }));
    assert(throwsInvalidArgument([] { Utils::getRandBetween(1.0, 1.0); }));
    assert(throwsInvalidArgument([] { Utils::randomWithProbability(-0.01); }));
    assert(throwsInvalidArgument([] { Utils::randomWithProbability(1.01); }));
}

void test_weight_initialization_uses_requested_scale() {
    Utils::setRandomSeed(9191);
    const double wide = Utils::getWeightRand(2.0);
    Utils::setRandomSeed(9191);
    const double narrow = Utils::getWeightRand(20.0);

    assert(std::isfinite(wide));
    assert(std::isfinite(narrow));
    assert(!near(narrow, 0.0, 1e-15));
    assert(near(wide / narrow, 10.0, 1e-8));

    assert(throwsInvalidArgument([] { Utils::getWeightRand(0.0); }));
    assert(throwsInvalidArgument([] { Utils::getWeightRand(-1.0); }));
}

}  // namespace

int main() {
    test_reverse_int();
    test_corrected_norm_semantics();
    test_squish_is_finite_direction_preserving_and_bounded();
    test_flatten_and_capsule_round_trip();
    test_binary_decode();
    test_rng_is_seedable_and_bounds_are_per_call();
    test_weight_initialization_uses_requested_scale();

    std::cout << "host Utils correctness tests passed" << std::endl;
    return 0;
}
