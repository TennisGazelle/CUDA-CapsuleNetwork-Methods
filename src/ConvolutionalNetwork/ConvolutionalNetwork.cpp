//
// Created by Daniel Lopez on 1/4/18.
//

#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <ConvolutionalNetwork/PoolingLayer.h>
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
    layers.push_back(new ConvolutionalLayer(28, 28, 10));
    layers.push_back(new ConvolutionalLayer(layers[0], 10));
    layers.push_back(new PoolingLayer(layers[1], MAX, 2, 5, 5));
    layers.push_back(new ConvolutionalLayer(layers[2], 10));
    layers.push_back(new PoolingLayer(layers[3], MAX, 2, 5, 5));

    finalLayers = new MultilayerPerceptron(layers[layers.size()-1]->getOutputSize1D(), 10, {10});
}

vector<double> ConvolutionalNetwork::loadImageAndGetOutput(int imageIndex, bool useTraining) {

}

void ConvolutionalNetwork::train() {

}

void ConvolutionalNetwork::writeToFile() {
    // TODO finish file name creation
    // make an fout
    ofstream fout;
    // output your own stuff
    finalLayers->writeToFile(fout);
    fout.close();
}