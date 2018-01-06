//
// Created by Daniel Lopez on 1/4/18.
//

#include <cstdlib>

#include "ConvolutionalNetwork/ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t iHeight, size_t iWidth, size_t numFilters, size_t fHeight, size_t fWidth)
        : filterHeight(fHeight), filterWidth(fWidth) {
    filters.resize(numFilters);
    // no parent, we are just setting inputs here (start of network)
    setInputDimension(iHeight, iWidth);
}

ConvolutionalLayer::ConvolutionalLayer(ICNLayer* pParent, size_t numFilters, size_t fHeight, size_t fWidth)
        : filterHeight(fHeight), filterWidth(fWidth) {
    filters.resize(numFilters);
    setParentLayer(pParent);
    setInputDimension(parent->outputHeight, parent->outputWidth);
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
                pixel = 0.0; // TODO Random function here on normal dist.interval [-1, 1]
            }
        }
    }

    // calculate what the output sizes should be in terms of images
    setOutputDimension(inputHeight - filterHeight, inputWidth - filterHeight);
}

void ConvolutionalLayer::calculateOutput() {
    for (int row = 0; row < inputHeight - filterHeight; row++) {
        for (int col = 0; col < inputWidth - filterWidth; col++) {
            for (int outputIndex = 0; outputIndex < filters.size(); outputIndex++) {
                outputMaps[outputIndex][row][col] = dotMatrixWithFilter(row, col, outputIndex);

                // ReLu
                if (outputMaps[outputIndex][row][col] < 0) {
                    outputMaps[outputIndex][row][col] = 0;
                }
            }
        }
    }
}

double ConvolutionalLayer::dotMatrixWithFilter(int beginRow, int beginCol, int filterIndex) const {
    double sum = 0.0;
    size_t count = 0;
    for (int row = beginRow; row < beginRow + filterHeight; row++) {
        for (int col = beginCol; col < beginCol + filterWidth; col++) {
            for (auto map : inputMaps) {
                sum += map[row][col] * filters[filterIndex][row - beginRow][col - beginCol];
            }
        }
    }

    return sum / double(count);
}