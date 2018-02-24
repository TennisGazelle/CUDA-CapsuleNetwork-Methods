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
    void runEpoch();
    void train();
    double tally(bool useTraining = true);
    vector<double> getErrorGradientVector(int targetLabel, const vector<double>& receivedOutput) const;
    vector<FeatureMap> backPropagate(const vector<double>& error);

    void batchUpdate();
    void writeToFile() const;
    void writeToFile(ofstream& fout) const;
    bool readFromFile(const string& filename);

private:
    // contents
    vector<ICNLayer*> layers;
    MultilayerPerceptron* finalLayers;
};


#endif //NEURALNETS_CONVOLUTIONALNETWORK_H
