//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_PERCEPTRON_H
#define NEURALNETS_PERCEPTRON_H

#include <vector>
#include "models/ActivationType.h"

using namespace std;

class Perceptron {
public:
    explicit Perceptron(ActivationType at = SIGMOID);
    void init(size_t numInputs);
    void populateFromFileRow(const vector<double> &line);
    double evaluate(const vector<double> &input) const;

    void selfAdjust(const double error, const vector<double> input);
    void adjustWeight(const double total);
    vector<double> reportDesire() const;
    double getWeightAt(int i) const;
    double getBias() const;

private:
    void adjustBias();
    void recordWeightAdjustment(const double error, const vector<double> prevInput);
    void calculateDesires();

    double bias;
    vector<double> weights;
    vector<double> weightAdjustment;
    vector<double> desire;
    ActivationType activationType;
};


#endif //NEURALNETS_PERCEPTRON_H
