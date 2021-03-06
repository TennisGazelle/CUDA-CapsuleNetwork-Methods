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

    void forwardPropagate();
    // returns the error gradient of the last layer (the input layer) just in case you want it
    vector<double> backPropagate(const vector<double>& errorGradient);
    void updateError();
    void updateWeights(const double total);
    vector<double> calculateErrorGradients(const vector<double> &previousErrorGradient);
    void outputLayerToFile(ofstream& fout) const;
private:
    void singleThreadedForwardPropagate();
    void multiThreadedForwardPropagate();
    void m_threading_forwardPropagate(int index);

    void singleThreadedBackPropagate(const vector<double>& errorGradient);
    void multiThreadedBackPropagate(const vector<double>& errorGradient);
    void m_threading_backPropagate(int index, double errorGradient);


    PerceptronLayer* parent;
    vector<Perceptron> perceptrons;
};


#endif //NEURALNETS_PERCEPTRONLAYER_H
