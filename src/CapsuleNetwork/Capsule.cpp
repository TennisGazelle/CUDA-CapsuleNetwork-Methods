//
// Created by daniellopez on 2/5/18.
//

#include <cassert>
#include <Utils.h>
#include "CapsuleNetwork/Capsule.h"

Capsule::Capsule(int iD, int oD, int inputs, int outputs) : inputDim(iD), outputDim(oD), numInputs(inputs), numOutputs(outputs) {
    init();
}

void Capsule::init() {
    weightMatricies.resize(numInputs);
    c.resize(numInputs);
    b.resize(numInputs);

    for (int i = 0; i < numInputs; i++) {
        b[i] = 1.0/double(numInputs);
        weightMatricies[i] = arma::mat(outputDim, inputDim, arma::fill::randu);
    }
    softmax();
}

void Capsule::softmax() {
    double sum_b_exps = 0.0;
    for (auto b_i : b) {
        sum_b_exps += exp(b_i);
    }
    for (int i = 0; i < numInputs; i++) {
        c[i] = exp(b[i] / sum_b_exps);
    }
}

arma::vec Capsule::calculateOutput(vector<arma::vec> inputs) const {
    // error check
    // we have as many inputs as we have weights for
    assert (inputs.size() == numInputs);
    // all inputs have the same dimensions
    for (auto const& v : inputs) {
        assert (v.size() == inputDim);
    }
    // TODO: also make sure the dimensions of the vec match the weightMatricies dimensions

    arma::vec sum(outputDim, arma::fill::zeros);
    for (size_t i = 0; i < weightMatricies.size(); i++) {
        // go multiply each by the weight matrix
        inputs[i] = weightMatricies[i] * inputs[i];
        // then with the c values,
        inputs[i] = c[i] * inputs[i];
        sum += inputs[i];
    }

    // activation function
    return Utils::squish(sum);
}