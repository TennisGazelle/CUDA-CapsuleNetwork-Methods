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
    weightMatrices.resize(numInputs);
    weightDeltas.resize(numInputs);
    c.resize(numInputs);
    b.resize(numInputs);

    for (int i = 0; i < numInputs; i++) {
        b[i] = 1.0/double(numInputs);
        weightMatrices[i] = arma::mat(outputDim, inputDim, arma::fill::randu);
        weightDeltas[i] = arma::mat(outputDim, inputDim, arma::fill::zeros);
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

vector<arma::vec> Capsule::backPropagate(const arma::vec &error) {
    vector<arma::vec> delta_u;
    for (int i = 0; i < numInputs; i++) {
        arma::vec delta_u_hat = trans(weightMatrices[i]) * error;
        delta_u_hat = c[i] * delta_u_hat;
        delta_u.push_back(delta_u_hat);

        // calculate your damn deltas
        weightDeltas[i] += (error * trans(prevInput[i]));
    }
    return delta_u;
}

arma::vec Capsule::forwardPropagate(const vector<arma::vec>& u) {
    // error check
    // we have as many inputs as we have weights for
    assert (u.size() == numInputs);
    // all inputs have the same dimensions
    for (auto const& v : u) {
        assert (v.size() == inputDim);
    }

    prevInput = u;

    output = routingAlgorithm();
    return output;
}

arma::vec Capsule::routingAlgorithm() {
    // calculate the u_hats
    vector<arma::vec> u_hat(prevInput.size());
    for (size_t i = 0; i < weightMatrices.size(); i++) {
        // go multiply each by the weight matrix
        u_hat[i] = weightMatrices[i] * prevInput[i];
    }

    // routing algorithm on page 3 starts here //
    // set all the b's to 0
    for (auto& b_val : b) {
        b_val = 0;
    }
    arma::vec v;
    for (int r = 0; r < numIterations; r++) {
        v = arma::vec(outputDim, arma::fill::zeros);
        softmax();

        // calculate s
        for (size_t i = 0; i < numInputs; i++) {
            // then with the c values,
            v += c[i] * u_hat[i];
        }

        // squash it
        v = Utils::squish(v);

        // update b's for everyone
        for (int i = 0; i < numInputs; i++) {
            b[i] += as_scalar(trans(u_hat[i]) * v);
        }
    }

    return v;
}

void Capsule::updateWeights() {
    for (int i = 0; i < numInputs; i++) {
        weightMatrices[i] += weightDeltas[i];
        weightDeltas[i].zeros();
    }
}

arma::vec Capsule::getOutput() const {
    return output;
}