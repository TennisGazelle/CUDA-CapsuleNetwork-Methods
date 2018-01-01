//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_MULTILAYERPERCEPTRON_H
#define NEURALNETS_MULTILAYERPERCEPTRON_H


#include "PerceptronLayer.h"
#include "MNISTReader.h"

class MultilayerPerceptron {
public:
    MultilayerPerceptron();
    void init(const string& possibleInputFilename = "");
    void run();
    vector<double> loadImageAndGetOutput(int imageIndex);
    void train();
    double tallyAndReportAccuracy();

    void writeToFile();
    bool readFromFile(const string& filename);
    void getLayerFromFile(ifstream& fin);

private:
    vector<PerceptronLayer> layers;
    MNISTReader reader;
};


#endif //NEURALNETS_MULTILAYERPERCEPTRON_H
