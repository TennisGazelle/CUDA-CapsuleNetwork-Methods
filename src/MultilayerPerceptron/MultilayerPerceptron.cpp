//
// Created by Daniel Lopez on 12/28/17.
//

#include <iostream>
#include <fstream>
#include "MultilayerPerceptron/MultilayerPerceptron.h"

MultilayerPerceptron::MultilayerPerceptron(size_t inputLayerSize, size_t outputLayerSize, vector<size_t> hiddenLayerSizes) {
    layerSizes.push_back(inputLayerSize);
    layerSizes.insert(layerSizes.end(), hiddenLayerSizes.begin(), hiddenLayerSizes.end());
    layerSizes.push_back(outputLayerSize);
}

void MultilayerPerceptron::init(const string& possibleInputFilename) {
    cout << "init..." << endl;
    reader.readMNISTData();

    // TODO - if there's a filename as parameter, read the neural net weights from a file
    if (!possibleInputFilename.empty() && readFromFile(possibleInputFilename)) {
        readFromFile(possibleInputFilename);
    } else {
        cout << "initializing network normally..." << endl;
        layers.reserve(layerSizes.size());

        layers.emplace_back(PerceptronLayer(layerSizes[0], layerSizes[1]));
        for (unsigned int i = 1; i < layerSizes.size(); i++) {
            layers.emplace_back(PerceptronLayer(&layers[i-1], layerSizes[i]));
        }

        for (auto& l : layers) {
            l.init();
        }
    }
}

void MultilayerPerceptron::run() {
    // train
    train();
    tallyAndReportAccuracy(false);
}

double MultilayerPerceptron::tallyAndReportAccuracy(bool useTraining) {
    cout << "tallying..." << endl;
    int numCorrectlyClassified = 0;

    auto tallyData = reader.trainingData;
    if (!useTraining)
        tallyData = reader.testingData;

    for (int i = 0; i < tallyData.size(); i++) {
        auto output = loadImageAndGetOutput(i, useTraining);

        size_t guess = 0;
        double highest = 0.0;
        for (int j = 0; j < output.size(); j++) {
            if (highest < output[j]) {
                highest = output[j];
                guess = j;
            }
//            cout << j << "--" << output[j] * 100 << endl;
        }
//        cout << "Guess  : " << guess << endl;
//        cout << "Actual : " << tallyData[i].getLabel() << endl;
//        cout << endl;
        if (guess == tallyData[i].getLabel()) {
            numCorrectlyClassified++;
        } else {
//            tallyData[i].print();
        }
    }
    cout << endl;
    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "Accuracy (out of " << tallyData.size() << ")       : " << double(numCorrectlyClassified)/double(tallyData.size()) * 100 << endl;
    return double(numCorrectlyClassified)/double(tallyData.size()) * 100;
}

vector<double> MultilayerPerceptron::loadImageAndGetOutput(int imageIndex, bool useTraining) {
    auto imageAsVector = reader.getTrainingImage(imageIndex).toVectorOfDoubles();
    if (!useTraining)
        imageAsVector = reader.getTestingImage(imageIndex).toVectorOfDoubles();

    layers[0].setInput(imageAsVector);
    for (auto& l : layers) {
        l.populateOutput();
    }

    return layers[layers.size()-1].getOutput();
}

void MultilayerPerceptron::train() {
    cout << "training with " << numTrainingEpochs << " epochs..." << endl;

    vector<double> history(numTrainingEpochs);
    for (unsigned int i = 0; i < numTrainingEpochs; i++) {
        cout << "=================" << endl;
        cout << "EPOCH ITERATION: " << i << endl;
        runEpoch();
        double accuracy = tallyAndReportAccuracy();
        tallyAndReportAccuracy(false);
        history[i] = accuracy;

        writeToFile();
    }
}

void MultilayerPerceptron::runEpoch(){
    for (int i = 0; i < reader.trainingData.size(); i++) {
        // set the "true" values
        vector<double> networkOutput = loadImageAndGetOutput(i);
        vector<double> desired(10, 0);
        desired[reader.trainingData[i].getLabel()] = 1.0;

        for (unsigned int j = 0; j < desired.size(); j++) {
            desired[j] = networkOutput[j] * (1-networkOutput[j]) * (desired[j] - networkOutput[j]);
        }

        // back-propagate!
        layers[layers.size()-1].backPropagate(desired);

        cout.precision(3);
        cout << fixed;
        if (!(i%5000) || i == reader.trainingData.size()-1) {
            cout << double(i) / double(reader.trainingData.size()) * 100 << "% of epoch done" << endl;
        }
    }
}

void MultilayerPerceptron::writeToFile() {
    // build the filename
    string outputfileName = "../bin/layer_weights/weights";
    for (auto& l : layerSizes) {
        outputfileName += "-" + to_string(l);
    }
    outputfileName += ".nnet";

    ofstream fout;
    fout.open(outputfileName);
    writeToFile(fout);
    fout.close();
}

void MultilayerPerceptron::writeToFile(ofstream &fout) {
    // output how many layers
    fout << layers.size() << endl;
    // for each layer
    for (int i = 0; i < layers.size(); i++) {
        // output the parent's layer's weights then nodes
        layers[i].outputLayerToFile(fout);
    }
}

bool MultilayerPerceptron::readFromFile(const string& filename) {
    ifstream fin(filename);
    if (!fin.good()) {
        cerr << "input file for network invalid or missing" << endl;
        return false;
    }

    int numLayers = 0;
    fin >> numLayers;

    layers.clear();
    for (int i = 0; i < numLayers; i++) {
        // get layer
        getLayerFromFile(fin);
    }

    fin.close();
    return true;
}

void MultilayerPerceptron::getLayerFromFile(ifstream &fin) {
    // get the first two, which define the input and outputs of the layer
    size_t layerInputSize, layerOutputSize;
    fin >> layerInputSize >> layerOutputSize;

    PerceptronLayer layer(layerInputSize, layerOutputSize);

    if (!layers.empty()) {
        layer.setParent(&layers[layers.size()-1]);
    }
    layer.init();

    // populate the layer
    vector< vector<double> > layerWeights;
    layerWeights.reserve(layerOutputSize);

    for (unsigned int r = 0; r < layerOutputSize; r++) {
        layerWeights.emplace_back(vector<double>(layerInputSize+1));
        for (unsigned int c = 0; c < layerInputSize+1; c++) {   // overshoot for bias
            fin >> layerWeights[r][c];
        }
    }
    layer.prePopulateLayer(layerWeights);

    layers.push_back(layer);
}