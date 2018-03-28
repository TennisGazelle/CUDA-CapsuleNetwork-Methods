//
// Created by daniellopez on 2/5/18.
//

#include <cassert>
#include <Utils.h>
#include <Config.h>
#include "CapsuleNetwork/Capsule.h"

void Capsule::init(int iD, int oD, int inputs, int outputs) {
    inputDim = iD;
    outputDim = oD;
    numInputs = inputs;
    numOutputs = outputs;

    weightMatrices.resize(numInputs);
    weightDeltas.resize(numInputs);
    weightVelocities.resize(numInputs);
    c.resize(numInputs);
    b.resize(numInputs);

    for (int i = 0; i < numInputs; i++) {
        b[i] = 0.0;
        weightMatrices[i] = arma::mat(outputDim, inputDim, arma::fill::randn);
        weightDeltas[i] = arma::mat(outputDim, inputDim, arma::fill::zeros);
        weightVelocities[i] = arma::mat(outputDim, inputDim, arma::fill::zeros);
    }
    softmax();
}

void Capsule::softmax() {
    long double sum_b_exps = 0.0;
    for (auto b_i : b) {
        if (!isnan(b_i) && !isinf(sum_b_exps + exp(b_i))) {
            sum_b_exps += exp(b_i);
        }
    }
    for (int i = 0; i < numInputs; i++) {
        c[i] = double(exp(b[i]) / sum_b_exps);
        if (isnan(c[i])) {
            cerr << "c[i] got nan" << endl;
            cerr << "       b[i]: " << b[i] << endl;
            cerr << "  exp(b[i]): " << exp(b[i]) << endl;
            cerr << " sum_b_exps: " << sum_b_exps << endl;
            exit(1);
        }
    }
}

vector<arma::vec> Capsule::backPropagate(const arma::vec &error) {
    vector<arma::vec> delta_u(numInputs);
    for (int i = 0; i < numInputs; i++) {
        arma::vec delta_u_hat = trans(weightMatrices[i]) * error;
        delta_u[i] = c[i] * delta_u_hat;

        // calculate your damn deltas
        weightDeltas[i] -= error * trans(prevInput[i]);
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
    for (auto& _b : b) {
        _b = 0.0;
    }

    arma::vec v;
    for (int r = 0; r < numIterations; r++) {
        softmax();

        // calculate s
        v = arma::vec(outputDim, arma::fill::zeros);
        for (size_t i = 0; i < numInputs; i++) {
            // then with the c values,
            v += c[i] * u_hat[i];
        }

        // squash it
        v = Utils::squish(v);

        // update b's for everyone
        for (int i = 0; i < numInputs; i++) {
            auto dot_product = arma::dot(u_hat[i], v);
            if (isnan(dot_product)) {
                sleep(1);
                cerr << "the dot product between the following two vectors is nan" << endl;
                u_hat[i].print("u_hat[i]");
                v.print("v");
                exit(1);
            }
            b[i] += dot_product;
        }
    }
    return v;
}

void Capsule::updateWeights() {
    for (int i = 0; i < numInputs; i++) {
        weightVelocities[i] = 0.9 * weightVelocities[i] + 0.1 * weightDeltas[i];
//        weightMatrices[i] += weightVelocities[i];
        weightMatrices[i] += weightDeltas[i];
        weightDeltas[i].zeros();
    }
}

arma::vec Capsule::getOutput() const {
    return output;
}