#include <CapsuleNetwork/BackpropUtils.h>

#include <armadillo>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool throwsInvalidArgument(void (*fn)()) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_accumulates_by_primary_capsule_index() {
    std::vector<arma::vec> accumulated(3, arma::vec(2, arma::fill::zeros));

    std::vector<arma::vec> first(3, arma::vec(2));
    first[0] = arma::vec({1.0, 10.0});
    first[1] = arma::vec({2.0, 20.0});
    first[2] = arma::vec({3.0, 30.0});

    std::vector<arma::vec> second(3, arma::vec(2));
    second[0] = arma::vec({4.0, 40.0});
    second[1] = arma::vec({5.0, 50.0});
    second[2] = arma::vec({6.0, 60.0});

    accumulatePrimaryCapsuleError(accumulated, first);
    accumulatePrimaryCapsuleError(accumulated, second);

    assert(arma::approx_equal(accumulated[0], arma::vec({5.0, 50.0}), "absdiff", 0.0));
    assert(arma::approx_equal(accumulated[1], arma::vec({7.0, 70.0}), "absdiff", 0.0));
    assert(arma::approx_equal(accumulated[2], arma::vec({9.0, 90.0}), "absdiff", 0.0));
}

void mismatchedVectorCount() {
    std::vector<arma::vec> accumulated(2, arma::vec(2, arma::fill::zeros));
    std::vector<arma::vec> contribution(3, arma::vec(2, arma::fill::zeros));
    accumulatePrimaryCapsuleError(accumulated, contribution);
}

void mismatchedVectorDimension() {
    std::vector<arma::vec> accumulated(2, arma::vec(2, arma::fill::zeros));
    std::vector<arma::vec> contribution(2, arma::vec(2, arma::fill::zeros));
    contribution[1] = arma::vec(3, arma::fill::zeros);
    accumulatePrimaryCapsuleError(accumulated, contribution);
}

void test_shape_mismatches_are_rejected() {
    assert(throwsInvalidArgument(mismatchedVectorCount));
    assert(throwsInvalidArgument(mismatchedVectorDimension));
}

}  // namespace

int main() {
    test_accumulates_by_primary_capsule_index();
    test_shape_mismatches_are_rejected();
    std::cout << "host capsule backprop reduction tests passed" << std::endl;
    return 0;
}
