//
// Created by Daniel Lopez on 1/4/18.
//

#include <cstdlib>
#include <Utils.h>

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

    // calculate what the output sizes should be in terms of images
    setOutputDimension(filters.size(), inputHeight - filterHeight, inputWidth - filterHeight);
}

void ConvolutionalLayer::calculateOutput() {
    for (int outputRow = 0; outputRow < inputHeight - filterHeight; outputRow++) {
        for (int outputCol = 0; outputCol < inputWidth - filterWidth; outputCol++) {
            for (int outputIndex = 0; outputIndex < filters.size(); outputIndex++) {
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
    return;
}

double ConvolutionalLayer::dotMatrixWithFilter(int beginRow, int beginCol, int filterIndex) const {
    double sum = 0.0;
    size_t count = 0;

    for (auto map : inputMaps) {
        for (int row = beginRow; row < beginRow + filterHeight; row++) {
            for (int col = beginCol; col < beginCol + filterWidth; col++) {
                sum += map[row][col] * filters[filterIndex][row - beginRow][col - beginCol];
                count++;
            }
        }
    }

    return sum / double(count);
}