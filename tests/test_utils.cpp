#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <Utils.h>

#include <armadillo>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "test_helpers/near.hpp"
#include "test_helpers/seed.hpp"

using capsnet::test::near;

TEST_CASE("reverse_int") {
    const int input = 0x01020304;
    const int expected = 0x04030201;
    CHECK((Utils::reverseInt(input) == expected));
}

TEST_CASE("corrected_norm_semantics") {
    arma::vec v(2);
    v[0] = 3.0;
    v[1] = 4.0;

    CHECK((near(Utils::square_length(v), 25.0)));
    CHECK((near(Utils::length(v), 5.0)));

    arma::vec zero(3, arma::fill::zeros);
    CHECK((near(Utils::square_length(zero), 0.0)));
    CHECK((near(Utils::length(zero), 0.0)));
    CHECK((arma::approx_equal(Utils::safeNormalise(zero), zero, "absdiff", 0.0)));
    CHECK((arma::approx_equal(Utils::squish(zero), zero, "absdiff", 0.0)));
}

TEST_CASE("squish_is_finite_direction_preserving_and_bounded") {
    arma::vec v(2);
    v[0] = 3.0;
    v[1] = 4.0;

    const arma::vec squashed = Utils::squish(v);
    CHECK((squashed.n_elem == 2));
    CHECK((std::isfinite(squashed[0])));
    CHECK((std::isfinite(squashed[1])));
    CHECK((squashed[0] > 0.0));
    CHECK((squashed[1] > 0.0));
    CHECK((near(squashed[0] / squashed[1], 3.0 / 4.0)));
    CHECK((Utils::length(squashed) < 1.0));
}

TEST_CASE("flatten_and_capsule_round_trip") {
    std::vector<arma::vec> capsules(2, arma::vec(3));
    capsules[0][0] = 1.0;
    capsules[0][1] = 2.0;
    capsules[0][2] = 3.0;
    capsules[1][0] = 4.0;
    capsules[1][1] = 5.0;
    capsules[1][2] = 6.0;

    const std::vector<double> flat = Utils::getAsOneDim(capsules);
    CHECK((flat.size() == 6));

    const std::vector<arma::vec> rebuilt = Utils::asCapsuleVectors(3, 2, flat);
    CHECK((rebuilt.size() == capsules.size()));
    for (std::size_t i = 0; i < capsules.size(); ++i) {
        for (std::size_t j = 0; j < capsules[i].n_elem; ++j) {
            CHECK((near(rebuilt[i][j], capsules[i][j])));
        }
    }

    const std::vector<arma::vec> empty;
    CHECK((Utils::getAsOneDim(empty).empty()));
    CHECK((Utils::asCapsuleVectors(3, 0, {}).empty()));

    CHECK_THROWS_AS(Utils::asCapsuleVectors(3, 2, {1.0, 2.0, 3.0}), std::invalid_argument);
    CHECK_THROWS_AS(Utils::asCapsuleVectors(0, 2, {}), std::invalid_argument);
}

TEST_CASE("binary_decode") {
    CHECK((Utils::getBinaryAsInt({}) == 0));
    CHECK((Utils::getBinaryAsInt({false, false, false, false, false}) == 0));
    CHECK((Utils::getBinaryAsInt({true, false, false, false, false}) == 16));
    CHECK((Utils::getBinaryAsInt({true, true, true, true, true}) == 31));
    CHECK((Utils::getBinaryAsInt({true, false, true, false, true}) == 21));
    CHECK_THROWS_AS(Utils::getBinaryAsInt(std::vector<bool>(32, true)), std::invalid_argument);
}

TEST_CASE("rng_is_seedable_and_bounds_are_per_call") {
    capsnet::test::set_utils_seed(12345);
    const int intA = Utils::getRandBetween(2, 7);
    const double realA = Utils::getRandBetween(-2.0, -1.0);
    const bool boolA = Utils::randomWithProbability(0.25);

    capsnet::test::set_utils_seed(12345);
    CHECK((Utils::getRandBetween(2, 7) == intA));
    CHECK((near(Utils::getRandBetween(-2.0, -1.0), realA)));
    CHECK((Utils::randomWithProbability(0.25) == boolA));

    capsnet::test::set_utils_seed(7);
    for (int i = 0; i < 100; ++i) {
        const int value = Utils::getRandBetween(2, 7);
        CHECK((value >= 2));
        CHECK((value < 7));
    }

    for (int i = 0; i < 50; ++i) {
        const double firstRange = Utils::getRandBetween(-10.0, -9.0);
        const double secondRange = Utils::getRandBetween(100.0, 101.0);
        CHECK((firstRange >= -10.0 && firstRange < -9.0));
        CHECK((secondRange >= 100.0 && secondRange < 101.0));
    }

    CHECK_THROWS_AS(Utils::getRandBetween(2, 2), std::invalid_argument);
    CHECK_THROWS_AS(Utils::getRandBetween(1.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(Utils::randomWithProbability(-0.01), std::invalid_argument);
    CHECK_THROWS_AS(Utils::randomWithProbability(1.01), std::invalid_argument);
}

TEST_CASE("weight_initialization_uses_requested_scale") {
    capsnet::test::set_utils_seed(9191);
    const double wide = Utils::getWeightRand(2.0);
    capsnet::test::set_utils_seed(9191);
    const double narrow = Utils::getWeightRand(20.0);

    CHECK((std::isfinite(wide)));
    CHECK((std::isfinite(narrow)));
    CHECK((!near(narrow, 0.0, 1e-15)));
    CHECK((near(wide / narrow, 10.0, 1e-8)));

    CHECK_THROWS_AS(Utils::getWeightRand(0.0), std::invalid_argument);
    CHECK_THROWS_AS(Utils::getWeightRand(-1.0), std::invalid_argument);
}
