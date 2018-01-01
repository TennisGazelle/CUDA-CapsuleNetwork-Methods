//
// Created by Daniel Lopez on 12/28/17.
//

#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include "MultilayerPerceptron.h"

MultilayerPerceptron::MultilayerPerceptron() {

}

void MultilayerPerceptron::init(const string& possibleInputFilename) {
    cout << "init..." << endl;
    reader.readMNISTData();

    // TODO - if there's a filename as parameter, read the neural net weights from a file
    if (!possibleInputFilename.empty() && readFromFile(possibleInputFilename)) {
        readFromFile(possibleInputFilename);
    } else {
        cout << "initializing network normally..." << endl;
        layers.reserve(3);
        layers.emplace_back(PerceptronLayer(28*28, 16));
        layers.emplace_back(PerceptronLayer(&layers[0], 16));
        layers.emplace_back(PerceptronLayer(&layers[1], 10));

        for (auto& l : layers) {
            l.init();
        }
    }
}

void MultilayerPerceptron::run() {
    // train a bunch of times
}

double MultilayerPerceptron::tallyAndReportAccuracy() {
    cout << "tallying..." << endl;
    int numCorrectlyClassified = 0;
    for (int i = 0; i < reader.images.size(); i++) {
        auto output = loadImageAndGetOutput(i);

        size_t guess = 0;
        double highest = 0.0;
        for (int j = 0; j < output.size(); j++) {
            if (highest < output[j]) {
                highest = output[j];
                guess = j;
            }
//            cout << j << "--" << output[j] << endl;
        }
//        cout << "Guess : " << guess << endl;
//        cout << "Actual:" << reader.images[i].getLabel() << endl;
        if (guess == reader.images[i].getLabel()) {
            numCorrectlyClassified++;
        }
//        cout << endl;
    }
    cout << endl;
    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "Accuracy (out of " << reader.images.size() << ")       : " << double(numCorrectlyClassified)/double(reader.images.size()) * 100 << endl;
    return double(numCorrectlyClassified)/double(reader.images.size()) * 100;
}

vector<double> MultilayerPerceptron::loadImageAndGetOutput(int imageIndex) {
    auto imageAsVector = reader.getImage(imageIndex).toVectorOfDoubles();

    layers[0].setInput(imageAsVector);
    layers[0].populateOutput();
    layers[1].populateOutput();
    layers[2].populateOutput();

    return layers[layers.size()-1].getOutput();
}

void MultilayerPerceptron::train() {
    cout << "training..." << endl;
    for (int i = 0; i < reader.images.size(); i++) {
        // set the "true" values
        vector<double> networkOutput = loadImageAndGetOutput(i);
        vector<double> desired(10, 0);
        desired[reader.images[i].getLabel()] = 1.0;

        for (unsigned int j = 0; j < desired.size(); j++) {
            desired[j] = networkOutput[j] * (1-networkOutput[j]) * (desired[j] - networkOutput[j]);
        }

        // back-propagate!
        layers[layers.size()-1].backPropagate(desired);

        if (!(i%20000) || i == reader.images.size()-1) {
            cout << double(i) / double(reader.images.size()) << "% training done" << endl;
        }
    }
}

void MultilayerPerceptron::writeToFile() {
    // build the filename
    string outputfileName = "../bin/layer_weights/weights";
    outputfileName += "-" + to_string(layers[0].getInputSize());
    for (auto& l : layers) {
        outputfileName += "-" + to_string(l.getOutputSize());
    }
    outputfileName += ".nnet";

    ofstream fout;
    fout.open(outputfileName);
    // output how many layers
    fout << layers.size() << endl;
    // for each layer
    for (int i = 0; i < layers.size(); i++) {
        // output the parent's layer's weights then nodes
        layers[i].outputLayerToFile(fout);
    }
    fout.close();
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