//
// Created by Daniel Lopez on 1/4/18.
//

#include <cassert>
#include "ConvolutionalNetwork/ICNLayer.h"

void ICNLayer::setInput(const vector<FeatureMap> &input) {
    inputMaps = input;
}

void ICNLayer::setParentLayer(ICNLayer *parentLayer) {
    // error check
    assert (parentLayer != nullptr);
    parent = parentLayer;
}

void ICNLayer::setInputDimension(size_t numInputs, size_t height, size_t width) {
    inputHeight = height;
    inputWidth = width;

    inputMaps.resize(numInputs);
    for (auto& map : inputMaps) {
        map.setSize(inputHeight, inputWidth);
    }
}

void ICNLayer::setOutputDimension(size_t numOutputs, size_t height, size_t width) {
    outputHeight = height;
    outputWidth = width;

    // size the outputs according to the channels
    outputMaps.resize(numOutputs);
    for (auto& map : outputMaps) {
        map.setSize(outputHeight, outputWidth);
    }
}

size_t ICNLayer::getOutputSize1D() const {
    return outputHeight * outputWidth;
}

vector<double> ICNLayer::getOutputAsOneDimensional() const {
    vector<double> oneDOutput;
    oneDOutput.reserve(outputMaps.size() * outputHeight * outputWidth);

    for (auto& map : outputMaps) {
        auto temp = map.toOneDim();
        oneDOutput.insert(oneDOutput.end(), temp.begin(), temp.end());
    }
}

void ICNLayer::process() {
    collectInput();
    calculateOutput();
}

void ICNLayer::collectInput() {
    // set my input from my parent's output
    if (parent != nullptr) {
        assert (parent->outputHeight == inputHeight);
        assert (parent->outputWidth  == inputWidth);
        inputMaps = parent->outputMaps;
    }
}