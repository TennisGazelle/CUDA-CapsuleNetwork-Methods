//
// Created by Daniel Lopez on 12/28/17.
//

#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "Utils.h"

using namespace std;

namespace {

mt19937& sharedGenerator() {
    static mt19937 gen(random_device{}());
    return gen;
}

}  // namespace

void Utils::setRandomSeed(std::uint32_t seed) {
    sharedGenerator().seed(seed);
}

int Utils::getRandBetween(int lowerBound, int upperBound) {
    if (lowerBound >= upperBound) {
        throw invalid_argument("integer random bounds require lowerBound < upperBound");
    }
    uniform_int_distribution<int> distribution(lowerBound, upperBound - 1);
    return distribution(sharedGenerator());
}

double Utils::getRandBetween(double lowerBound, double upperBound) {
    if (!(lowerBound < upperBound)) {
        throw invalid_argument("real random bounds require lowerBound < upperBound");
    }
    uniform_real_distribution<double> distribution(lowerBound, upperBound);
    return distribution(sharedGenerator());
}

double Utils::getWeightRand(double n) {
    if (!(n > 0.0) || !isfinite(n)) {
        throw invalid_argument("weight initialization scale n must be finite and positive");
    }

    // Historical intent: approximately 99.7% of a normal distribution lies
    // inside [-2.4/n, 2.4/n]. Three standard deviations therefore span the
    // historical half-width, giving sigma = 0.8/n.
    const double standardDeviation = (2.4 / n) / 3.0;
    normal_distribution<double> distribution(0.0, standardDeviation);
    return distribution(sharedGenerator());
}

int Utils::reverseInt(int i) {
    unsigned char
        c1 = i & 255,
        c2 = (i >> 8) & 255,
        c3 = (i >> 16) & 255,
        c4 = (i >> 24) & 255;

    return ((int) c1 << 24) +
           ((int) c2 << 16) +
           ((int) c3 << 8) +
           ((int) c4);
}

double Utils::square_length(const arma::vec &vn) {
    double sum = 0.0;
    for (const auto& v : vn) {
        sum += v * v;
    }
    return sum;
}

double Utils::length(const arma::vec &vn) {
    return sqrt(square_length(vn));
}

double Utils::getSquashDerivativeLength(const arma::vec &input) {
    const double l = length(input);
    return (2.0 * l) / pow(l * l + 1.0, 2.0);
}

arma::vec Utils::squish(const arma::vec &input) {
    const double lengthSquared = Utils::square_length(input);
    if (lengthSquared == 0.0) {
        return input;
    }

    const double squishingScalar = lengthSquared / (1.0 + lengthSquared);
    return squishingScalar * safeNormalise(input);
}

arma::vec Utils::safeNormalise(arma::vec input) {
    const double l = length(input);
    if (l <= numeric_limits<double>::epsilon()) {
        return input;
    }
    return input / l;
}

vector<double> Utils::getAsOneDim(const vector<arma::vec> &input) {
    size_t totalElements = 0;
    for (const auto& capsule : input) {
        totalElements += capsule.n_elem;
    }

    vector<double> result;
    result.reserve(totalElements);
    for (const auto& capsule : input) {
        for (arma::uword j = 0; j < capsule.n_elem; ++j) {
            result.push_back(capsule[j]);
        }
    }
    return result;
}

vector<arma::vec> Utils::asCapsuleVectors(int dim, int numVectors, const vector<double> &data) {
    if (dim <= 0 || numVectors < 0) {
        throw invalid_argument("capsule shape requires dim > 0 and numVectors >= 0");
    }

    const size_t expectedSize = static_cast<size_t>(dim) * static_cast<size_t>(numVectors);
    if (data.size() != expectedSize) {
        throw invalid_argument("capsule data size does not match requested shape");
    }

    vector<arma::vec> result(static_cast<size_t>(numVectors), arma::vec(static_cast<arma::uword>(dim), arma::fill::zeros));
    for (int v = 0; v < numVectors; ++v) {
        for (int d = 0; d < dim; ++d) {
            result[static_cast<size_t>(v)][static_cast<arma::uword>(d)] =
                data[static_cast<size_t>(v * dim + d)];
        }
    }
    return result;
}

bool Utils::randomWithProbability(double prob) {
    if (prob < 0.0 || prob > 1.0 || !isfinite(prob)) {
        throw invalid_argument("probability must be finite and in [0, 1]");
    }
    bernoulli_distribution distribution(prob);
    return distribution(sharedGenerator());
}

int Utils::getBinaryAsInt(const std::vector<bool> &subset) {
    if (subset.empty()) {
        return 0;
    }
    if (subset.size() > 31) {
        throw invalid_argument("binary subset is too large for signed int decoding");
    }

    unsigned int value = 0;
    for (bool bit : subset) {
        value = (value << 1U) | (bit ? 1U : 0U);
    }
    return static_cast<int>(value);
}
