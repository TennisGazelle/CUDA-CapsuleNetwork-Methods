//
// Created by Daniel Lopez on 1/4/18.
//

#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <ConvolutionalNetwork/PoolingLayer.h>
#include <iostream>
#include <ProgressBar.h>
#include <Config.h>
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
    layers.push_back(new ConvolutionalLayer(Config::inputHeight, Config::inputHeight, 16, 10, 10));
//    layers.push_back(new PoolingLayer(layers[0], MAX, 2, 2, 2));
//    layers.push_back(new ConvolutionalLayer(layers[0], 32));
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

    return finalLayers->loadInputAndGetOutput(
            layers[layers.size()-1]->getOutputAsOneDimensional()
    );
}

void ConvolutionalNetwork::train() {
    vector<double> history;
    history.reserve(Config::getInstance()->getNumEpochs());
    for (size_t i = 0; i < Config::getInstance()->getNumEpochs(); i++) {
        cout << "EPOCH ITERATION:" << i << endl;
        runEpoch();
        history.push_back(tally(false));
        writeToFile();

        cout << endl;
    }

    for (int i = 0; i < history.size(); i++) {
        cout << i << "," << history[i] << endl;
    }
    cout << "DONE!" << endl;
}

void ConvolutionalNetwork::writeToFile() const {
    cout << "saving..." << endl;
    string outputfileName = "../bin/layer_weights/cnn";
    for (auto ptr : layers) {
        if (typeid(*ptr) == typeid(ConvolutionalLayer)) {
            outputfileName += "-c" + to_string(ptr->outputMaps.size());
        } else if (typeid(*ptr) == typeid(PoolingLayer)) {
            outputfileName += "-p";
        }
    }
    for (auto num : finalLayers->getSizes()) {
        outputfileName += "-" + to_string(num);
    }
    outputfileName += ".nnet";

    // make an fout
    ofstream fout;
    // output your own stuff
    fout.open(outputfileName);
    writeToFile(fout);
    fout.close();
}

void ConvolutionalNetwork::writeToFile(ofstream &fout) const {
    // layer metadata
    fout << layers.size();
    for (auto ptr : layers) {
        if (typeid(*ptr) == typeid(ConvolutionalLayer)) {
            fout << " c";
        } else if (typeid(*ptr) == typeid(PoolingLayer)) {
            fout << " p";
        }
    }
    if (finalLayers) {
        fout << " mlp";
    }
    fout << endl;

    // layer data
    for (auto ptr : layers) {
        ptr->outputLayerToFile(fout);
    }

    if (finalLayers) {
        finalLayers->writeToFile(fout);
    }
}

bool ConvolutionalNetwork::readFromFile(const string &filename) {
    // go through, read and give to layers
    // if I reach MLP, give that to the finaLayers
    // finish
    return true;
}

void ConvolutionalNetwork::runEpoch() {
    cout << "training ... " << endl;
    const unsigned int batchSize = 100;
    const bool useTraining = false;
    vector<Image> tallyData;

    if (useTraining) {
        tallyData = MNISTReader::getInstance()->trainingData;
    } else {
        tallyData = MNISTReader::getInstance()->testingData;
    }

    ProgressBar progressBar(tallyData.size());
    for (int i = 0; i < tallyData.size(); i++) {
        auto image = tallyData[i];

        // forward propagation
        layers[0]->setInput({image.toFeatureMap()});
        for (auto& l : layers) {
            l->process();
        }
        vector<double> mlpOutput = finalLayers->loadInputAndGetOutput(
                layers[layers.size()-1]->getOutputAsOneDimensional()
        );

        // calc error for back-propagation (target loss func.)
        vector<double> error = getErrorGradientVector(image.getLabel(), mlpOutput);

        backPropagate(error);

        // back-propagation
        auto mlpError = finalLayers->backPropagateError(error);
        layers[layers.size()-1]->backPropagate(FeatureMap::toFeatureMaps(
                layers[layers.size()-1]->outputHeight,
                layers[layers.size()-1]->outputWidth,
                mlpError
        ));

        if (i % batchSize == 0 || i == tallyData.size()-1) {
            batchUpdate();
        }
        progressBar.updateProgress(i);
    }
}

vector<double> ConvolutionalNetwork::getErrorGradientVector(int targetLabel, const vector<double>& receivedOutput) const {
    vector<double> result(receivedOutput.size());
    for (int j = 0; j < receivedOutput.size(); j++) {
        double target = 0;
        if (j == targetLabel) {
            target = 1;
        }
        result[j] = receivedOutput[j] * (1-receivedOutput[j]) * (target - receivedOutput[j]);
    }
    return result;
}

void ConvolutionalNetwork::batchUpdate() {
    finalLayers->batchUpdate();
    layers[layers.size()-1]->updateError();
}

double ConvolutionalNetwork::tally(bool useTraining) {
    cout << "tallying ..." << endl;
    int numCorrectlyClassified = 0;
    vector<Image> tallyData;
    if (useTraining) {
        tallyData = MNISTReader::getInstance()->trainingData;
    } else {
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

vector<FeatureMap> ConvolutionalNetwork::backPropagate(const vector<double> &error) {
    auto mlpError = finalLayers->backPropagateError(error);
    layers[layers.size()-1]->backPropagate(FeatureMap::toFeatureMaps(
            layers[layers.size()-1]->outputHeight,
            layers[layers.size()-1]->outputWidth,
            mlpError
    ));
}