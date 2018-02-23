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
    vector<arma::mat> weightMatricies;
    vector<double> c;
    vector<double> b;

    int inputDim;
    int outputDim;
    int numInputs;
    int numOutputs;
};


#endif //NEURALNETS_CAPSULE_H
