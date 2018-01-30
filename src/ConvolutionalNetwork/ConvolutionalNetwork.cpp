//
// Created by Daniel Lopez on 1/4/18.
//

#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <ConvolutionalNetwork/PoolingLayer.h>
#include <iostream>
#include <ProgressBar.h>
#include "ConvolutionalNetwork/ConvolutionalNetwork.h"

ConvolutionalNetwork::~ConvolutionalNetwork() {
    for (auto& ptr : layers) {
        if (ptr) {
            delete ptr;
            ptr = nullptr;
        }
    }

    if (finalLayers) {
        delete finalLayers;
        finalLayers = nullptr;
    }
}

void ConvolutionalNetwork::init() {
    layers.push_back(new ConvolutionalLayer(28, 28, 20));
    layers.push_back(new PoolingLayer(layers[0], MAX, 2, 5, 5));
    layers.push_back(new ConvolutionalLayer(layers[1], 10));
    layers.push_back(new PoolingLayer(layers[2], MAX, 2, 5, 5));

    finalLayers = new MultilayerPerceptron(layers[layers.size()-1]->getOutputSize1D(), 10, {10});
    finalLayers->init();
}

vector<double> ConvolutionalNetwork::loadImageAndGetOutput(int imageIndex, bool useTraining) {
    auto image = MNISTReader::getInstance()->getTrainingImage(imageIndex).to2DImage();
    if (!useTraining)
        image = MNISTReader::getInstance()->getTestingImage(imageIndex).to2DImage();

    // sent as 'vector' of 2d images
    layers[0]->setInput({image});
    for (auto& l : layers) {
        l->process();
    }

    return finalLayers->loadInputAndGetOutput(layers[layers.size()-1]->getOutputAsOneDimensional());
}

void ConvolutionalNetwork::train() {
    vector<double> history;
    history.reserve(200);
    for (size_t i = 0; i < 200; i++) {
        cout << "EPOCH ITERATION:" << i << endl;
        runEpoch();
        history.push_back(tally(false));

        // write to File
    }

    for (int i = 0; i < history.size(); i++) {
        cout << i << "," << history[i] << endl;
    }
    cout << "DONE!" << endl;
}

void ConvolutionalNetwork::writeToFile() {
    // TODO finish file name creation
    // make an fout
    ofstream fout;
    // output your own stuff
    finalLayers->writeToFile(fout);
    fout.close();
}

void ConvolutionalNetwork::runEpoch() {
    cout << "training ... " << endl;
    auto data = MNISTReader::getInstance()->testingData;

    ProgressBar progressBar(data.size());
    for (int i = 0; i < data.size(); i++) {
        vector<double> networkOutput = loadImageAndGetOutput(i, false);
        vector<double> desired(10, 0);
        desired[data[i].getLabel()] = 1.0;
        for (unsigned int j = 0; j < desired.size(); j++) {
            desired[j] = networkOutput[j] * (1-networkOutput[j]) * (desired[j] - networkOutput[j]);
        }

        // back-propagation
        // send this to the back fo the mlp and get the desired stuff back
        auto mlpError = finalLayers->backPropagateError(desired);

        layers[layers.size()-1]->backPropagate(FeatureMap::toFeatureMaps(
                layers[layers.size()-1]->outputHeight,
                layers[layers.size()-1]->outputWidth,
                mlpError
        ));

        progressBar.updateProgress(i);
    }
}

double ConvolutionalNetwork::tally(bool useTraining) {
    cout << "tallying ..." << endl;
    int numCorrectlyClassified = 0;
    auto tallyData = MNISTReader::getInstance()->trainingData;
    if (!useTraining) {
        tallyData = MNISTReader::getInstance()->testingData;
    }

    for (int i = 0; i < tallyData.size(); i++) {
        auto output = loadImageAndGetOutput(i, useTraining);
        size_t guess = 0;
        double highest = 0.0;
        for (size_t j = 0; j < output.size(); j++) {
            if (highest < output[j]) {
                guess = j;
                highest = output[j];
            }
        }
        if (guess == tallyData[i].getLabel()) {
            numCorrectlyClassified++;
        }
    }
    cout << endl;
    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "Accuracy (out of " << tallyData.size() << ")       : " << double(numCorrectlyClassified)/double(tallyData.size()) * 100 << endl;
    return double(numCorrectlyClassified)/double(tallyData.size()) * 100;
}