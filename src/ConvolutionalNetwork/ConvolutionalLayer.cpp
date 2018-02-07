//
// Created by Daniel Lopez on 1/4/18.
//

#include <cstdlib>
#include <Utils.h>
#include <cassert>
#include <iostream>
#include <iomanip>

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
                pixel = Utils::getRandBetween(-1, 1);
//                pixel = Utils::getWeightRand(28*28);
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

    const static double learningRate = 0.01;
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

void ConvolutionalLayer::printKernel(int channel) {
    cout << "filter: " << channel << endl;
    cout << setprecision(3);
    cout << fixed;
    for (int r = 0; r < filterHeight; r++) {
        for (int c = 0; c < filterWidth; c++) {
            cout << filters[channel][r][c] << "\t";
        }
        cout << endl;
    }
}

void ConvolutionalLayer::printOutput(int channel) {
    for (int r = 0; r < outputHeight; r++) {
        for (int c = 0; c < outputWidth; c++) {
            if (outputMaps[channel][r][c] < 100) {
                cout << "0";
            }
            if (outputMaps[channel][r][c] < 10) {
                cout << "0";
            }
            cout << outputMaps[channel][r][c] << "\t";
        }
        cout << endl;
    }
}