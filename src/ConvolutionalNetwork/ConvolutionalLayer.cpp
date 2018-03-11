//
// Created by Daniel Lopez on 1/4/18.
//

#include <cstdlib>
#include <Utils.h>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <Config.h>

#include "ConvolutionalNetwork/ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t iHeight, size_t iWidth, size_t numFilters, size_t fHeight, size_t fWidth)
        : filterDepth(1), filterHeight(fHeight), filterWidth(fWidth) {
    filters.resize(numFilters);
    // no parent, we are just setting inputs here (start of network)
    setInputDimension(1, iHeight, iWidth);  // no RGB either, so just 1 input channel
    init();
}

ConvolutionalLayer::ConvolutionalLayer(ICNLayer* pParent, size_t numFilters, size_t fHeight, size_t fWidth)
        : filterDepth(pParent->outputMaps.size()), filterHeight(fHeight), filterWidth(fWidth) {
    filters.resize(numFilters);
    setParentLayer(pParent);
    setInputDimension(parent->outputMaps.size(), parent->outputHeight, parent->outputWidth);
    init();
}

void ConvolutionalLayer::init() {
    for (auto& filter : filters) {
        filter.init(filterDepth, filterHeight, filterWidth);
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
                double dotProduct = dotMatrixWithFilter(outputRow, outputCol, outputIndex);
                // ReLu
                if (dotProduct > 0) {
                    outputMaps[outputIndex][outputRow][outputCol] = dotProduct;
                } else {
                    outputMaps[outputIndex][outputRow][outputCol] = 0;
                }
//                // Tangental Activation
//                double e_z = exp(dotProduct);
//                double e_zn = exp(-dotProduct);
//                outputMaps[outputIndex][outputRow][outputCol] = (e_z - e_zn) / (e_z + e_zn);
            }
        }
    }
}

vector<FeatureMap> ConvolutionalLayer::backPropagate(const vector<FeatureMap> &errorGradient) {
    // error check
    assert (errorGradient.size() == outputMaps.size());
    double highestDesiredInput = 0.0;

    // clear the nudges desired for the previous layers
    for (auto& desired : inputDesiredChange) {
        desired.clearOut();
    }

    // for every weight per output node (which is shared amongst them)
    for (size_t outputChannel = 0; outputChannel < outputMaps.size(); outputChannel++) {
        // for every weight, go through all the output values that use it
        for (size_t outputRow = 0; outputRow < outputHeight; outputRow++) {
            for (size_t outputCol = 0; outputCol < outputWidth; outputCol++) {
                for (size_t filterChannel = 0; filterChannel < filterDepth; filterChannel++) {
                    for (size_t filterRow = 0; filterRow < filterHeight; filterRow++) {
                        for (size_t filterCol = 0; filterCol < filterWidth; filterCol++) {
                            double adjustment = Config::getInstance()->getLearningRate() * inputMaps[filterChannel][outputRow+filterRow][outputCol+filterCol] * errorGradient[outputChannel][outputRow][outputCol];
                            filterAdjustments[outputChannel][filterChannel][filterRow][filterCol] -= adjustment;
                            inputDesiredChange[filterChannel][outputRow+filterRow][outputCol+filterCol] += (filters[outputChannel][filterChannel][filterRow][filterCol] > 0) ? 1 : -1;
                            highestDesiredInput = max(highestDesiredInput, inputDesiredChange[filterChannel][outputRow+filterRow][outputCol+filterCol]);
                        }
                    }
                }
            }
        }
    }

    // go through that new desired output (which is the same dimensions as the input)
    // and calculate the error function for it (from ConvolutionalNetwork::runEpoch())
    for (size_t ch = 0; ch < inputMaps.size(); ch++) {
        for (size_t r = 0; r < inputHeight; r++) {
            for (size_t c = 0; c < inputWidth; c++) {
                double target = inputDesiredChange[ch][r][c] / highestDesiredInput;
//                double target = max(0.0, inputDesiredChange[ch][r][c]);
                inputDesiredChange[ch][r][c] = inputMaps[ch][r][c] * (1 - inputMaps[ch][r][c]) * (target - inputMaps[ch][r][c]);
            }
        }
    }
    if (parent != nullptr) {
        // be recursive
        parent->backPropagate(inputDesiredChange);
    }
    return inputDesiredChange;
}

double ConvolutionalLayer::dotMatrixWithFilter(int beginRow, int beginCol, int filterIndex) const {
    double sum = 0.0;
    size_t count = 0;

    for (int channel = 0; channel < filterDepth; channel++) {
        for (int row = 0; row < filterHeight; row++) {
            for (int col = 0; col < filterWidth; col++) {
                sum += inputMaps[channel][row + beginRow][col + beginCol] * filters[filterIndex][channel][row][col];
                count++;
            }
        }
    }

    return sum/count;
}

void ConvolutionalLayer::printKernel(int channel) {
    cout << "filter: " << channel << endl;
    cout << setprecision(3);
    cout << fixed;
    for (int ch = 0; ch < filterDepth; ch++) {
        cout << "filter depth : " << ch << endl;
        for (int r = 0; r < filterHeight; r++) {
            for (int c = 0; c < filterWidth; c++) {
                cout << filters[channel][ch][r][c] << "\t";
            }
            cout << endl;
        }
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

void ConvolutionalLayer::outputLayerToFile(ofstream& fout) const {
    // specs of convolutional layers
    fout << filters.size() << " " << filterHeight << " " << filterWidth << endl;
    for (int filterIndex = 0; filterIndex < filters.size(); filterIndex++) {
        for (int ch = 0; ch < filterDepth; ch++) {
            for (int r = 0; r < filterHeight; r++) {
                for (int c = 0; c < filterWidth; c++) {
                    fout << filters[filterIndex][ch][r][c] << "\t";
                }
                fout << endl;
            }
            fout << endl;
        }
    }
}

void ConvolutionalLayer::updateError() {
    // apply the adjustments
    for (int i = 0; i < filters.size(); i++) {
        filters[i] = filters[i] + filterAdjustments[i];
        filterAdjustments[i].clearOut();
    }

    if (parent != nullptr) {
        parent->updateError();
    }
}