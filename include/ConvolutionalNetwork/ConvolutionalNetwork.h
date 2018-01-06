//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_CONVOLUTIONALNETWORK_H
#define NEURALNETS_CONVOLUTIONALNETWORK_H


#include <MultilayerPerceptron/MultilayerPerceptron.h>
#include "ICNLayer.h"

class ConvolutionalNetwork {
public:
    ~ConvolutionalNetwork();

    void init();
    vector<double> loadImageAndGetOutput(int imageIndex, bool useTraining = true);
    void train();

    void writeToFile();

private:
    // contents
    vector<ICNLayer*> layers;
    MultilayerPerceptron* finalLayers;

    MNISTReader reader;
};


#endif //NEURALNETS_CONVOLUTIONALNETWORK_H
