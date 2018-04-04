//
// Created by Daniel Lopez on 12/28/17.
//

#include <cassert>
#include <thread>
#include "MultilayerPerceptron/PerceptronLayer.h"

PerceptronLayer::PerceptronLayer(size_t pInputSize, size_t numNodes) {
    parent = nullptr;
    inputSize = pInputSize;
    outputSize = numNodes;
}

PerceptronLayer::PerceptronLayer(PerceptronLayer *parentLayer, size_t numNodes) {
    parent = parentLayer;
    inputSize = parentLayer->outputSize;
    outputSize = numNodes;
}

void PerceptronLayer::init() {
    input.resize(inputSize);
    output.resize(outputSize);

    perceptrons.resize(outputSize);
    for (auto& p : perceptrons) {
        p.init(inputSize);
    }
}

void PerceptronLayer::prePopulateLayer(const vector<vector<double> > &weightMatrix) {
    // for each row
    for (unsigned int i = 0; i < weightMatrix.size(); i++) {
        // first number is the bias for the node
        // all numbers are subsequent weights in order
        perceptrons[i].populateFromFileRow(weightMatrix[i]);
    }
}

void PerceptronLayer::setParent(PerceptronLayer* parentLayer) {
    parent = parentLayer;
    inputSize = parentLayer->outputSize;
}

void PerceptronLayer::forwardPropagate() {
    // if i have a parent, save their output to my input
    if (parent != nullptr) {
        // error checking
        assert (inputSize == parent->outputSize);
        input = parent->output;
    }
    singleThreadedForwardPropagate();
//    if (Config::getInstance()->multithreaded) {
//        multiThreadedForwardPropagate();
//    } else {
//    }
    for (unsigned int i = 0; i < outputSize; i++) {
        output[i] = perceptrons[i].evaluate(input);
    }
}

void PerceptronLayer::singleThreadedForwardPropagate() {
    for (unsigned int i = 0; i < outputSize; i++) {
        output[i] = perceptrons[i].evaluate(input);
    }
}

void PerceptronLayer::multiThreadedForwardPropagate() {
    // find good number
    thread workers[outputSize];
    for (unsigned int i = 0; i < outputSize; i++) {
        workers[i] = thread(&PerceptronLayer::m_threading_forwardPropagate, this, i);
    }
    for (auto& w : workers) {
        w.join();
    }
}

void PerceptronLayer::m_threading_forwardPropagate(int index) {
    output[index] = perceptrons[index].evaluate(input);
}

vector<double> PerceptronLayer::backPropagate(const vector<double>& errorGradient) {
    // error check
    assert(errorGradient.size() == outputSize);

    singleThreadedBackPropagate(errorGradient);
//    if (Config::getInstance()->multithreaded) {
//        multiThreadedBackPropagate(errorGradient);
//    } else {
//    }

    vector<double> previousErrorGradient = calculateErrorGradients(errorGradient);

    if (parent != nullptr) {
        return parent->backPropagate(previousErrorGradient);
    }
    return previousErrorGradient;
}

void PerceptronLayer::singleThreadedBackPropagate(const vector<double> &errorGradient) {
    // collect the nudges reported by every perceptron
    for (unsigned int pIndex = 0; pIndex < perceptrons.size(); pIndex++) {
        perceptrons[pIndex].selfAdjust(errorGradient[pIndex], input);
    }
}

void PerceptronLayer::multiThreadedBackPropagate(const vector<double> &errorGradient) {
    thread workers[perceptrons.size()];
    for (unsigned int i = 0; i < perceptrons.size(); i++) {
        workers[i] = thread(&PerceptronLayer::m_threading_backPropagate, this, i, errorGradient[i]);
    }
    for (auto& w : workers) {
        w.join();
    }
}

void PerceptronLayer::m_threading_backPropagate(int index, double errorGradient) {
    perceptrons[index].selfAdjust(errorGradient, input);
}

vector<double> PerceptronLayer::calculateErrorGradients(const vector<double> &previousErrorGradient) {
    vector<double> errors(inputSize);
    for (unsigned int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
        double weightedSum = 0.0;
        for (unsigned int outputIndex = 0; outputIndex < outputSize; outputIndex++) {
            weightedSum += perceptrons[outputIndex].getWeightAt(inputIndex) * previousErrorGradient[outputIndex];
        }
        errors[inputIndex] = input[inputIndex] * (1-input[inputIndex]) * weightedSum;
    }
    return errors;
}

void PerceptronLayer::updateWeights(const double total) {
    // tell perceptrons to update weight
    for (unsigned int i = 0; i < perceptrons.size(); i++) {
        perceptrons[i].adjustWeight(total);
    }
}

void PerceptronLayer::outputLayerToFile(ofstream& fout) const {
    // how many weights and nodes are in this layer
    fout << inputSize << " " << outputSize << endl;
    // output the 'bias' of node_i, then all weights for for node_i
    for (unsigned int i = 0; i < perceptrons.size(); i++) {
        fout << perceptrons[i].getBias() << "\t\t";
        for (unsigned int j = 0; j < inputSize; j++) {
            fout << perceptrons[i].getWeightAt(j) << "\t";
        }
        fout << endl;
    }
}

void PerceptronLayer::updateError() {
    updateWeights(60000.0);
    // if there's a parent, do it too
    if (parent != nullptr) {
        parent->updateError();
    }
}