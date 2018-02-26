//
// Created by daniellopez on 2/5/18.
//

#ifndef NEURALNETS_CAPSULE_H
#define NEURALNETS_CAPSULE_H

#include <armadillo>
#include <vector>

using namespace std;

class Capsule {
public:
    Capsule(int iD, int oD, int inputs, int outputs);
    void init();
    void softmax();
    // squishification function
    arma::vec calculateOutput(vector<arma::vec> inputs) const;
    vector<arma::vec> backPropagate(const arma::vec& error);
    arma::vec forwardPropagate(const vector<arma::vec>& u);
    void updateWeights();

private:
    arma::vec routingAlgorithm();

    vector<arma::mat> weightMatrices, weightDeltas;
    vector<double> c;
    vector<double> b;

    int inputDim;
    int outputDim;
    int numInputs;
    int numOutputs;

    const int numIterations = 3;
    vector<arma::vec> prevInput;
};


#endif //NEURALNETS_CAPSULE_H
