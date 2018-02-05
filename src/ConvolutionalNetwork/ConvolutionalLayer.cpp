//
// Created by Daniel Lopez on 1/4/18.
//

#include <cstdlib>
#include <Utils.h>
#include <cassert>
#include <iostream>

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
                pixel = Utils::getWeightRand(28*28); // TODO Random function here on normal dist.interval [-1, 1]
            }
        }
    }
    filterAdjustments = filters;
    for (auto& fAdj : filterAdjustments) {
        fAdj.clearOut();
    }
    inputDesiredChange = inputMaps;
    for (auto& iDC : inputDesiredChange) {
        iDC.clearOut();
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
    for (auto& desired : inputDesiredChange) {
        desired.clearOut();
    }

    const static double learningRate = 0.1;
    const static double momentum = 0.85;

    // for every weight per output node (which is shared amongst them)
    for (size_t outputChannel = 0; outputChannel < outputMaps.size(); outputChannel++) {
        for (size_t outputRow = 0; outputRow < outputHeight; outputRow++) {
            for (size_t outputCol = 0; outputCol < outputWidth; outputCol++) {
                for (size_t filterRow = 0; filterRow < filterHeight; filterRow++) {
                    for (size_t filterCol = 0; filterCol < filterWidth; filterCol++) {
                        for (size_t inputChannel = 0; inputChannel < inputMaps.size(); inputChannel++) {
                            // adjust the weights by going through the outputs and have each one voice their output
                            double adjustment = (learningRate * inputMaps[inputChannel][filterRow+outputRow][filterCol + outputCol] * errorGradient[outputChannel][outputRow][outputCol])
                                                * (momentum * filters[outputChannel][filterRow][filterCol]);
                            filterAdjustments[outputChannel][filterRow][filterCol] += adjustment;

                            // calculate the desires that every weight says the "input" should be for the gradient
                            inputDesiredChange[inputChannel][filterRow+outputRow][filterCol+outputCol] += (filters[outputChannel][filterRow][filterCol] > 1) ? 1 : 0;
                        }
                    }
                }
            }
        }
        // update the weighs themselves
        // THINK: worth overloading '+=' for Filters?
        filters[outputChannel] = filters[outputChannel] + filterAdjustments[outputChannel];
    }

    // be recursive
    if (parent != nullptr) {
        vector<FeatureMap> expandedPrevErrorGradient = inputMaps;
        // for each input
        for (size_t inputChannel = 0; inputChannel < inputMaps.size(); inputChannel++) {
            expandedPrevErrorGradient[inputChannel].clearOut();
            for (size_t inputRow = 0; inputRow < inputHeight-filterHeight; inputRow++) {
                for (size_t inputCol = 0; inputCol < inputWidth-filterWidth; inputCol++) {
                    // the weight times the output error gradient for this input node
                    double weightedSum = 0.0;
                    // go through the output nodes associated with this one
                    for (size_t filterChannel = 0; filterChannel < filters.size(); filterChannel++) {
                        for (size_t filterRow = 0; filterRow < filterHeight; filterRow++) {
                            for (size_t filterCol = 0; filterCol < filterWidth; filterCol++) {
                                if ((inputRow + filterRow > 0 && inputRow + filterRow < outputHeight) &&
                                    (inputCol + filterCol > 0 && inputCol + filterCol < outputWidth)) {
                                    weightedSum += filters[filterChannel][filterRow][filterCol] * errorGradient[filterChannel][inputRow+filterRow][inputCol+filterCol];
                                }
                            }
                        }
                    }
                    expandedPrevErrorGradient[inputChannel][inputRow][inputCol] = inputMaps[inputChannel][inputRow][inputChannel] * (1-inputMaps[inputChannel][inputRow][inputChannel]) * (weightedSum);
                }
            }
        }

        parent->backPropagate(expandedPrevErrorGradient);
    }
}

double ConvolutionalLayer::dotMatrixWithFilter(int beginRow, int beginCol, int filterIndex) const {
    double sum = 0.0;
    size_t count = 0;

    for (auto& map : inputMaps) {
        for (int row = beginRow; row < beginRow + filterHeight; row++) {
            for (int col = beginCol; col < beginCol + filterWidth; col++) {
                if (row < inputHeight && col < inputWidth) {
                    sum += map[row][col] * filters[filterIndex][row - beginRow][col - beginCol];
                }
                count++;
            }
        }
    }

    return sum;
}

void ConvolutionalLayer::mapError(FeatureMap& prevErrorGradient, const vector<FeatureMap>& errorGradient, size_t beginRow, size_t beginCol) {
    for (size_t filterChannel = 0; filterChannel < filters.size(); filterChannel++) {
        for (size_t row = beginRow; row < beginRow + filterHeight; row++) {
            for (size_t col = beginCol; col < beginCol + filterWidth; col++) {
                double prevError = 0;
                if (beginRow < outputHeight && beginCol < outputWidth) {
                    prevError = errorGradient[0][beginRow][beginCol];
                }
                double w_ab = filters[filterChannel][row-beginRow][col-beginCol];

                prevErrorGradient[beginRow][beginCol] += prevError * w_ab;
            }
        }
    }
}

void ConvolutionalLayer::updateFilterAdj(size_t filterIndex, size_t filterRow, size_t filterCol, const vector<FeatureMap>& error) {
    const static double learningRate = 0.1;
    const static double momentum = 0.85;

    // one weight is used for multiple outputs, get the sum of these b_i's and delta_i's
    double sum = 0.0;
    for (size_t inputIndex = 0; inputIndex < inputMaps.size(); inputIndex++) {
        for (size_t r = 0; r < outputHeight - filterHeight + 1; r++) {
            for (size_t c = 0; c < outputWidth - filterWidth + 1; c++) {
                sum += error[inputIndex][r][c] * inputMaps[inputIndex][r+filterRow][c+filterCol];
            }
        }
    }

    filterAdjustments[filterIndex][filterRow][filterCol] += (learningRate * sum) + (momentum * filterAdjustments[filterIndex][filterRow][filterCol]);
}