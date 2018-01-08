//
// Created by Daniel Lopez on 1/4/18.
//

#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <ConvolutionalNetwork/PoolingLayer.h>
#include <iostream>
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
    // TODO: get 2D image from MNIST reader
    auto imageAsVector = MNISTReader::getInstance()->getTrainingImage(imageIndex).to2DImage();
    if (!useTraining)
        imageAsVector = MNISTReader::getInstance()->getTestingImage(imageIndex).to2DImage();

    layers[0]->setInput({imageAsVector});
    for (auto& l : layers) {
        l->process();
    }

    return finalLayers->loadInputAndGetOutput(layers[layers.size()-1]->getOutputAsOneDimensional());
}

void ConvolutionalNetwork::train() {
    auto output = loadImageAndGetOutput(0);

    for (int i = 0; i < output.size(); i++) {
        cout << i << "--" << output[i] * 100 << endl;
    }
    cout << endl << endl;

    runEpoch();

    output = loadImageAndGetOutput(0);

    for (int i = 0; i < output.size(); i++) {
        cout << i << "--" << output[i] * 100 << endl;
    }
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
    vector<double> networkOutput = loadImageAndGetOutput(0);
    vector<double> desired(10, 0);
    desired[MNISTReader::getInstance()->trainingData[0].getLabel()] = 1.0;

    for (unsigned int i = 0; i < desired.size(); i++) {
        desired[i] = networkOutput[i] * (1-networkOutput[i]) * (desired[i] - networkOutput[i]);
    }

    // back-propagation
    // send this to the back fo the mlp and get the desired stuff back
    auto mlpError = finalLayers->backPropagateError(desired);

    layers[layers.size()-1]->backPropagate(FeatureMap::toFeatureMaps(
            layers[layers.size()-1]->outputHeight,
            layers[layers.size()-1]->outputWidth,
            mlpError
    ));
}