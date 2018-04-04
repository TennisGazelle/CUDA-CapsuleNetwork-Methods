//
// Created by Daniel Lopez on 12/28/17.
//

#include <Utils.h>
#include <cassert>
#include <Config.h>
#include "MultilayerPerceptron/Perceptron.h"

Perceptron::Perceptron() : bias(0.0) {
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
    // activation function from all the weights
    switch (Config::getInstance()->at) {
        case SIGMOID:
            return 1.0 / (1.0 + exp(-sum));
        default:
            cerr << "Perceptron activation type not defined" << endl;
            exit(1);
    }
}

void Perceptron::selfAdjust(const double error, const vector<double> input) {
    adjustBias(error);
    recordWeightAdjustment(error, input);
    calculateDesires();
}

void Perceptron::adjustBias(const double error) {
//    bias += 0.001 * (error);
}

void Perceptron::recordWeightAdjustment(const double error, const vector<double> prevInput) {
    // change w_i by factor of a_i
    for (unsigned int i = 0; i < weights.size(); i++) {
        // by a factor of the previous input for this neuron...
        double adjustment = (Config::getInstance()->getLearningRate() * prevInput[i] * error);
        weightAdjustment[i] += adjustment;
    }
}

void Perceptron::adjustWeight(const double total) {
    for (unsigned int i = 0; i < weights.size(); i++) {
        weights[i] += weightAdjustment[i];
        weightAdjustment[i] = 0.0;
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