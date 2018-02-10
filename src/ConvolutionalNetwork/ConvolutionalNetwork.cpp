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
    layers.push_back(new ConvolutionalLayer(28, 28, 32));
//    layers.push_back(new PoolingLayer(layers[0], MAX, 2, 2, 2));
//    layers.push_back(new ConvolutionalLayer(layers[1], 64));
//    layers.push_back(new PoolingLayer(layers[2], MAX, 2, 5, 5));

    finalLayers = new MultilayerPerceptron(layers[layers.size()-1]->getOutputSize1D(), 10, {16});
    finalLayers->init();
}

vector<double> ConvolutionalNetwork::loadImageAndGetOutput(int imageIndex, bool useTraining) {
    FeatureMap image;
    if (useTraining) {
        image = MNISTReader::getInstance()->getTrainingImage(imageIndex).toFeatureMap();
    } else {
        image = MNISTReader::getInstance()->getTestingImage(imageIndex).toFeatureMap();
    }

    // sent as 'vector' of 2d images
    layers[0]->setInput({image});
    for (auto& l : layers) {
        l->process();
    }

    return finalLayers->loadInputAndGetOutput(layers[layers.size()-1]->getOutputAsOneDimensional());
}

void ConvolutionalNetwork::train() {
    vector<double> history;
    history.reserve(500);
    for (size_t i = 0; i < 500; i++) {
        cout << "EPOCH ITERATION:" << i << endl;
        runEpoch();
        history.push_back(tally(false));
        cout << endl;
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

    ProgressBar progressBar(MNISTReader::getInstance()->testingData.size());
    for (int i = 0; i < MNISTReader::getInstance()->testingData.size(); i++) {
        auto image = MNISTReader::getInstance()->testingData[i];

        layers[0]->setInput({image.toFeatureMap()});
        layers[0]->calculateOutput();
        vector<double> mlpOutput = finalLayers->loadInputAndGetOutput(layers[0]->getOutputAsOneDimensional());

        vector<double> error(mlpOutput.size());
        for (int j = 0; j < mlpOutput.size(); j++) {
            double target = 0;
            if (j == image.getLabel()) {
                target = 1;
            }
            error[j] = mlpOutput[j] * (1-mlpOutput[j]) * (target - mlpOutput[j]);
        }

        // back-propagation
        // send this to the back fo the mlp and get the desired stuff back
        auto mlpError = finalLayers->backPropagateError(error);

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
    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "Accuracy (out of " << tallyData.size() << ")       : " << double(numCorrectlyClassified)/double(tallyData.size()) * 100 << endl;
    return double(numCorrectlyClassified)/double(tallyData.size()) * 100;
}