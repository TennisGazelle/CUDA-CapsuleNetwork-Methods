//
// Created by Daniel Lopez on 12/28/17.
//

#include <Utils.h>
#include <assert.h>
#include <cmath>
#include "Perceptron.h"

Perceptron::Perceptron(ActivationType at) : activationType(at), bias(0.0) {
}

void Perceptron::init(size_t numInputs) {
    weights.resize(numInputs, 0.0);
    weightAdjustment.resize(numInputs, 0.0);
    desire.resize(numInputs, 0.0);
    // make each a random number
    for (auto& w : weights) {
//        w = Utils::getRandBetween(-1, 1);
        w = Utils::getWeightRand(28*28);
    }
}

void Perceptron::populateFromFileRow(const vector<double> &line) {
    bias = line[0];
    assert (weights.size() == line.size()-1);

    // go through and populate weights
    for (unsigned int i = 0; i < weights.size(); i++) {
        weights[i] = line[i+1]; // skip one for the bias
    }
}

double Perceptron::evaluate(const vector<double> &input) const {
    // assuming that the size of the input is equal to as many weights as I have...
    assert(input.size() == weights.size());

    // sum and sigmoid it!
    double sum = bias;
    for (unsigned int i = 0; i < input.size(); i++) {
        sum += weights[i] * input[i];
    }
    return 1.0 / (1.0 + exp(-sum));
}

void Perceptron::selfAdjust(const double error, const vector<double> input) {
    adjustBias();
    recordWeightAdjustment(error, input);
    calculateDesires();
}

void Perceptron::adjustBias() {

}

void Perceptron::recordWeightAdjustment(const double error, const vector<double> prevInput) {
    // change w_i by factor of a_i
    const static double learningRate = .001;
    const static double momentum = 0.9;

    for (unsigned int i = 0; i < weights.size(); i++) {
        // by a factor of the previous input for this neuron...
        double adjustment = (learningRate * prevInput[i] * error) + (momentum * weightAdjustment[i]);
        weightAdjustment[i] = adjustment;
    }
}

void Perceptron::adjustWeight(const double total) {
    for (unsigned int i = 0; i < weights.size(); i++) {
        weights[i] += weightAdjustment[i];
//        weightAdjustment[i] = 0.0;
    }
}

void Perceptron::calculateDesires() {
    // change a_i by a factor of w_i
    for (unsigned int i = 0; i < desire.size(); i++) {
        // by a factor of the weight of this connection..
        if (weights[i] > 0.0)
            desire[i] = 1;
        else
            desire[i] = 0;
    }
}

vector<double> Perceptron::reportDesire() const {
    return desire;
}

double Perceptron::getWeightAt(int i) const {
    return weights[i];
}

double Perceptron::getBias() const {
    return bias;
}