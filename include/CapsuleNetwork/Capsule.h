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
    Capsule(int d, int numInputs);
    void init();
    void softmax();
    // squishification function
    arma::vec routingIteration(vector<arma::vec> inputs) const;
    arma::vec squish(arma::vec input) const;
    vector<arma::mat> weightMatricies;
    vector<double> c;
    vector<double> b;

    int dim;
    int expectedNumInputs;
};


#endif //NEURALNETS_CAPSULE_H
