#include <Utils.h>

#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool near(double lhs, double rhs, double tolerance = 1e-10) {
    return std::abs(lhs - rhs) <= tolerance;
}

void test_reverse_int() {
    const int input = 0x01020304;
    const int expected = 0x04030201;
    assert(Utils::reverseInt(input) == expected);
}

void test_historical_norm_semantics() {
    arma::vec v(2);
    v[0] = 3.0;
    v[1] = 4.0;

    // Characterization test: the historical implementation adds EPSILON in
    // square_length and then adds EPSILON again in length. This is documented
    // as an audit item; do not "correct" the expected values in this test
    // without explicitly changing compatibility semantics.
    const double expected_square = 25.0 + EPSILON;
    const double expected_length = std::sqrt(25.0 + 2.0 * EPSILON);

    assert(near(Utils::square_length(v), expected_square));
    assert(near(Utils::length(v), expected_length));
}

void test_squish_is_finite_and_direction_preserving() {
    arma::vec v(2);
    v[0] = 3.0;
    v[1] = 4.0;

    arma::vec squashed = Utils::squish(v);
    assert(squashed.n_elem == 2);
    assert(std::isfinite(squashed[0]));
    assert(std::isfinite(squashed[1]));

    // Positive scalar rescaling should preserve direction/ratiometric order.
    assert(squashed[0] > 0.0);
    assert(squashed[1] > 0.0);
    assert(near(squashed[0] / squashed[1], 3.0 / 4.0));
    assert(Utils::length(squashed) < 1.01);
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
}

void test_binary_decode() {
    assert(Utils::getBinaryAsInt({false, false, false, false, false}) == 0);
    assert(Utils::getBinaryAsInt({true, false, false, false, false}) == 16);
    assert(Utils::getBinaryAsInt({true, true, true, true, true}) == 31);
    assert(Utils::getBinaryAsInt({true, false, true, false, true}) == 21);
}

void test_integer_random_bounds() {
    std::srand(7);
    for (int i = 0; i < 100; ++i) {
        const int value = Utils::getRandBetween(2, 7);
        assert(value >= 2);
        assert(value < 7);
    }
}

}  // namespace

int main() {
    test_reverse_int();
    test_historical_norm_semantics();
    test_squish_is_finite_and_direction_preserving();
    test_flatten_and_capsule_round_trip();
    test_binary_decode();
    test_integer_random_bounds();

    std::cout << "host Utils characterization tests passed" << std::endl;
    return 0;
}
