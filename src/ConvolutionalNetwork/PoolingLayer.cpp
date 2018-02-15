//
// Created by Daniel Lopez on 1/4/18.
//

#include <cmath>
#include <cassert>
#include "ConvolutionalNetwork/PoolingLayer.h"

PoolingLayer::PoolingLayer(ICNLayer* pParent, PoolingType pt, size_t sSize, size_t wHeight, size_t wWidth)
        : windowHeight(wHeight), windowWidth(wWidth), strideSize(sSize), poolingType(pt) {
    // calculate the output sizes
    setParentLayer(pParent);

    setInputDimension(parent->outputMaps.size(), parent->outputHeight, parent->outputWidth);
    init();
}

void PoolingLayer::init() {
    // size the outputs according to the channels
    setOutputDimension(inputMaps.size(), size_t(ceil(double(inputHeight)/double(strideSize))), size_t(ceil(double(inputWidth)/double(strideSize))));
}

void PoolingLayer::calculateOutput() {
    // pooling layers have to have parents
    assert (parent != nullptr);

    for (size_t channel = 0; channel < parent->outputMaps.size(); channel++) {
        for (size_t rowBegin = 0, rowIndex = 0; rowBegin < inputHeight; rowBegin += strideSize, rowIndex++) {
            for (size_t colBegin = 0, colIndex = 0; colBegin < inputWidth; colBegin += strideSize, colIndex++) {
                outputMaps[channel][rowIndex][colIndex] = findPoolValue(rowBegin, colBegin, channel);
            }
        }
    }
}

void PoolingLayer::backPropagate(const vector<FeatureMap> &errorGradient) {
    // do stuff if it's
    assert (errorGradient.size() == outputMaps.size());

    // there is no weight adjustment, so just map the error to the input maps
    vector<FeatureMap> prevErrorGradient = inputMaps;
    for (auto& map : prevErrorGradient) {
        map.clearOut();
    }

    // window through
    for (size_t channel = 0; channel < parent->outputMaps.size(); channel++) {
        for (size_t rowBegin = 0, rowIndex = 0; rowBegin < inputHeight; rowBegin += strideSize, rowIndex++) {
            for (size_t colBegin = 0, colIndex = 0; colBegin < inputWidth; colBegin += strideSize, colIndex++) {
                if (poolingType == MAX) {
                    // for max pooling, just map the error to the map value (the other values get no error)
                    auto coords = returnCoordinatesOfHighest(rowBegin, colBegin, channel);
                    prevErrorGradient[channel][coords.first][coords.second] = errorGradient[channel][rowIndex][colIndex];
                } else if (poolingType == MEAN) {
                    // for mean pooling, give every cell in the window 1/(wH * wW)-th of the error
                    // TODO: implement mean pooling later
                }
            }
        }
    }

    // give this error gradient to the previous guys
    parent->backPropagate(prevErrorGradient);
}

pair<size_t, size_t> PoolingLayer::returnCoordinatesOfHighest(size_t rowBegin, size_t colBegin, size_t channel) {
    pair<size_t, size_t> coordinatesOfHighest = {rowBegin, colBegin};
    double highest = 0.0;

    for (size_t row = rowBegin; row < rowBegin + windowHeight; row++) {
        for (size_t col = colBegin; col < colBegin + windowWidth; col++) {
            if (row < windowHeight && col < windowWidth && highest < inputMaps[channel][row][col]) {
                highest = inputMaps[channel][row][col];
                coordinatesOfHighest = {row, col};
            }
        }
    }

    return coordinatesOfHighest;
}

double PoolingLayer::findPoolValue(size_t rowBegin, size_t colBegin, size_t channel) {
    double sum = 0.0;
    double highest = 0.0;
    size_t count = 0;
    for (size_t row = rowBegin; row < rowBegin + windowHeight; row++) {
        for (size_t col = colBegin; col < colBegin + windowWidth; col++) {
            if (row < windowHeight && col < windowWidth) {
                highest = max(highest, inputMaps[channel][row][col]);
                sum += inputMaps[channel][row][col];
                count++;
            }
        }
    }

    if (poolingType == MAX) {
        return highest;
    } else {
        return sum/double(count);
    }
}

void PoolingLayer::updateError() {
    // this layer has no weights to update with mini-batching, just skipping
    assert(parent != nullptr);

    parent->updateError();
}