//
// Created by daniellopez on 2/5/18.
//

#include <cassert>
#include <Utils.h>
#include "CapsuleNetwork/Capsule.h"

Capsule::Capsule(int d, int numInputs) : dim(d), expectedNumInputs(numInputs) {
    init();
}

void Capsule::init() {
    weightMatricies.reserve(expectedNumInputs);
    c.reserve(expectedNumInputs);
    b.reserve(expectedNumInputs);

    for (int i = 0; i < expectedNumInputs; i++) {
        b.emplace_back(1.0/double(expectedNumInputs));
        weightMatricies.emplace_back(arma::mat(dim, dim, arma::fill::randu));
    }
}

void Capsule::softmax() {
    double sum_b_exps = 0.0;
    for (auto b_i : b) {
        sum_b_exps += exp(b_i);
    }
    for (int i = 0; i < expectedNumInputs; i++) {
        c[i] = exp(b[i] / sum_b_exps);
    }
}

arma::vec Capsule::routingIteration(vector<arma::vec> inputs) const {
    // error check
    // we have as many inputs as we have weights for
    assert (inputs.size() == expectedNumInputs);
    // all inputs have the same dimensions
    auto vec_size = inputs[0].size();
    for (auto const& v : inputs) {
        assert (v.size() == vec_size);
    }
    // TODO: also make sure the dimensions of the vec match the weightMatricies dimensions

    arma::vec sum(vec_size, arma::fill::zeros);
    for (size_t i = 0; i < weightMatricies.size(); i++) {
        // go multiply each by the weight matrix
        inputs[i] = weightMatricies[i] * inputs[i];
        // then with the c values,
        inputs[i] = c[i] * inputs[i];
        sum += inputs[i];
    }

    // activation function
    return squish(sum);
}

arma::vec Capsule::squish(arma::vec input) const {
    auto lengthSquared = Utils::square_length(input);
    auto squishingScalar = lengthSquared / (1 + lengthSquared);
    return squishingScalar * normalise(input, 1);
}