//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_PERCEPTRONLAYER_H
#define NEURALNETS_PERCEPTRONLAYER_H

#include <fstream>
#include "ILayer.h"
#include "Perceptron.h"

class PerceptronLayer : public ILayer {
public:
    PerceptronLayer(size_t pInputSize, size_t numNodes);
    PerceptronLayer(PerceptronLayer* parentLayer, size_t numNodes);

    void init();
    void prePopulateLayer(const vector< vector<double> > &weightMatrix);
    void setParent(PerceptronLayer* parent);

    void populateOutput();
    void backPropagate(const vector<double> errorGradient);
    void updateWeights(const double total);
    vector<double> calculateErrorGradients(const vector<double> &previousErrorGradient);
    void outputLayerToFile(ofstream& fout) const;

    int getInputSize() const;
    int getOutputSize() const;

private:
    PerceptronLayer* parent;

    vector<Perceptron> perceptrons;
    vector<double> sumNudges;
};


#endif //NEURALNETS_PERCEPTRONLAYER_H
