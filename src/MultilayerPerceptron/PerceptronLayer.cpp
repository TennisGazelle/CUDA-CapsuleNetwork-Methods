//
// Created by Daniel Lopez on 12/28/17.
//

#include <cassert>
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
    sumNudges.resize(inputSize);
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

void PerceptronLayer::populateOutput() {
    // if i have a parent, save their output to my input
    if (parent != nullptr) {
        // error checking
        assert (inputSize == parent->outputSize);
        input = parent->output;
    }
    for (unsigned int i = 0; i < outputSize; i++) {
        output[i] = perceptrons[i].evaluate(input);
    }
}

void PerceptronLayer::backPropagate(const vector<double> errorGradient) {
    // clear out the nudges
    for (auto& d : sumNudges)
        d = 0.0;
    // error check
    assert(errorGradient.size() == outputSize);

    // collect the nudges reported by every perceptron
    for (unsigned int pIndex = 0; pIndex < perceptrons.size(); pIndex++) {
        perceptrons[pIndex].selfAdjust(errorGradient[pIndex], input);
        auto nudge = perceptrons[pIndex].reportDesire();

        for (int i = 0; i < nudge.size(); i++) {
            sumNudges[i] += nudge[i];
        }
    }

    vector<double> previousErrorGradient = calculateErrorGradients(errorGradient);

    // now that I have the "error", tell the parent to do the same
    if (parent != nullptr) {
        parent->backPropagate(previousErrorGradient);
    }

    updateWeights(60000.0);
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
//    for (auto& p : perceptrons) { // THIS DOESN'T WORK, FIGURE OUT WHY
//        p.adjustWeight(total);
//    }
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

int PerceptronLayer::getInputSize() const {
    return (int) inputSize;
}

int PerceptronLayer::getOutputSize() const {
    return (int) outputSize;
}