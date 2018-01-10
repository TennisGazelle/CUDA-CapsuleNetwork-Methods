//
// Created by Daniel Lopez on 1/4/18.
//

#include <cstdlib>
#include <Utils.h>
#include <cassert>

#include "ConvolutionalNetwork/ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t iHeight, size_t iWidth, size_t numFilters, size_t fHeight, size_t fWidth)
        : filterHeight(fHeight), filterWidth(fWidth) {
    filters.resize(numFilters);
    // no parent, we are just setting inputs here (start of network)
    setInputDimension(1, iHeight, iWidth);
    init();
}

ConvolutionalLayer::ConvolutionalLayer(ICNLayer* pParent, size_t numFilters, size_t fHeight, size_t fWidth)
        : filterHeight(fHeight), filterWidth(fWidth) {
    filters.resize(numFilters);
    setParentLayer(pParent);
    setInputDimension(parent->outputMaps.size(), parent->outputHeight, parent->outputWidth);
    init();
}

void ConvolutionalLayer::init() {
    // height
    for (auto& filter : filters) {
        filter.resize(filterHeight);
        // width
        for (auto& row : filter) {
            row.resize(filterWidth);
            // depth
            for (auto& pixel : row) {
                pixel = Utils::getRandBetween(-1, 1); // TODO Random function here on normal dist.interval [-1, 1]
            }
        }
    }
    filterAdjustments = filters;
    for (auto& fAdj : filterAdjustments) {
        fAdj.clearOut();
    }

    // calculate what the output sizes should be in terms of images
    setOutputDimension(filters.size(), inputHeight - filterHeight, inputWidth - filterHeight);
}

void ConvolutionalLayer::calculateOutput() {
    for (int outputIndex = 0; outputIndex < filters.size(); outputIndex++) {
        for (int outputRow = 0; outputRow < inputHeight - filterHeight; outputRow++) {
            for (int outputCol = 0; outputCol < inputWidth - filterWidth; outputCol++) {
                outputMaps[outputIndex][outputRow][outputCol] = dotMatrixWithFilter(outputRow, outputCol, outputIndex);
                // ReLu
                if (outputMaps[outputIndex][outputRow][outputCol] < 0) {
                    outputMaps[outputIndex][outputRow][outputCol] = 0;
                }
            }
        }
    }
}

void ConvolutionalLayer::backPropagate(const vector<FeatureMap> &errorGradient) {
    // error check
    assert (errorGradient.size() == outputMaps.size());

    // map the error to the inputs (previous layer)
    vector<FeatureMap> prevErrorGradient = inputMaps;
    for (auto& map : prevErrorGradient) {
        map.clearOut();
    }

    // map the adjustment needed for each filter, per window
    for (size_t outputIndex = 0; outputIndex < filters.size(); outputIndex++) {
        for (size_t outputRow = 0; outputRow < outputHeight; outputRow++) {
            for (size_t outputCol = 0; outputCol < outputRow; outputCol++) {
                // redistribute the error gradient to all the weights
                mapError(prevErrorGradient, errorGradient[outputIndex][outputRow][outputCol], outputRow, outputCol);
            }
        }
    }

    // adjust the filters according to some learning rate
    for (size_t filterIndex = 0; filterIndex < filters.size(); filterIndex++) {
        for(size_t filterRow = 0; filterRow < filters[filterIndex].size(); filterRow++) {
            for (size_t filterCol = 0; filterCol < filters[filterIndex][filterRow].size(); filterCol++) {
                updateFilterAdj(filterIndex, filterRow, filterCol);
            }
        }

        // THINK: worth overloading '+=' for Filters?
        filters[filterIndex] = filters[filterIndex] + filterAdjustments[filterIndex];
    }

    // remap the error to the previous layer
    return;
}

double ConvolutionalLayer::dotMatrixWithFilter(int beginRow, int beginCol, int filterIndex) const {
    double sum = 0.0;
    size_t count = 0;

    for (auto& map : inputMaps) {
        for (int row = beginRow; row < beginRow + filterHeight; row++) {
            for (int col = beginCol; col < beginCol + filterWidth; col++) {
                sum += map[row][col] * filters[filterIndex][row - beginRow][col - beginCol];
                count++;
            }
        }
    }

    return sum / double(count);
}

void ConvolutionalLayer::mapError(vector<FeatureMap>& mapping, double error, size_t beginRow, size_t beginCol) {
    for (size_t inputChannel = 0; inputChannel < inputMaps.size(); inputChannel++) {
        for (size_t row = beginRow; row < beginRow + filterHeight; row++) {
            for (size_t col = beginCol; col < beginCol + filterWidth; col++) {
                double prevInput = inputMaps[inputChannel][row][col];
                mapping[inputChannel][row][col] += error * prevInput * (1-prevInput);
            }
        }
    }
}

void ConvolutionalLayer::updateFilterAdj(size_t filterIndex, size_t filterRow, size_t filterCol) {
    const static double learningRate = 0.001;
    const static double momentum = 0.9;

    // one weight is used for multiple outputs, get the sum of these b_i's and delta_i's

}