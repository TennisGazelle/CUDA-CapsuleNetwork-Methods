//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_MULTILAYERPERCEPTRON_H
#define NEURALNETS_MULTILAYERPERCEPTRON_H


#include "PerceptronLayer.h"
#include "MNISTReader.h"
#include "Config.h"

class MultilayerPerceptron {
public:
    MultilayerPerceptron(const Config& incomingConfig, size_t inputLayerSize, size_t outputLayerSize, vector<size_t> hiddenLayerSizes);
    void init(const string& possibleInputFilename = "");
    vector<double> loadImageAndGetOutput(int imageIndex, bool useTraining = true);
    vector<double> loadInputAndGetOutput(const vector<double>& input);
    void train();

    // TODO: break this function down to get the error inputs from the input fields...
    void runEpoch();
    vector<double> backPropagateError(const vector<double>& error);
    double tally(bool useTraining = true);

    void writeToFile();
    void writeToFile(ofstream &fout);
    bool readFromFile(const string& filename);
    bool readFromFile(ifstream &fin);
    bool getLayerFromFile(ifstream& fin);
    vector<size_t> getSizes() const;
    void batchUpdate();

private:
    vector<PerceptronLayer> layers;
    vector<size_t> layerSizes;
    Config config;
};


#endif //NEURALNETS_MULTILAYERPERCEPTRON_H
