//
// Created by Daniel Lopez on 1/4/18.
//

#include <cstdlib>
#include <Utils.h>
#include <cassert>
#include <iomanip>
#include <thread>

#include "ConvolutionalNetwork/ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t iHeight, size_t iWidth, size_t numFilters, size_t fHeight, size_t fWidth)
        : filterDepth(1), filterHeight(fHeight), filterWidth(fWidth) {
    filters.resize(numFilters);
    filterAdjustments.resize(numFilters);
    filterVelocities.resize(numFilters);
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
    for (int i = 0; i < filters.size(); i++) {
        filters[i].init(filterDepth, filterHeight, filterWidth);
        filterAdjustments[i].clearInit(filterDepth, filterHeight, filterWidth);
        filterVelocities[i].clearInit(filterDepth, filterHeight, filterWidth);
    }
    newErrorGradient = inputMaps;
    for (auto& iDC : newErrorGradient) {
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
                outputMaps[outputIndex][outputRow][outputCol] = max(0.0, dotProduct);
                outputMaps[outputIndex][outputRow][outputCol] = dotProduct;
//                // Tangental Activation
//                double e_z = exp(dotProduct);
//                double e_zn = exp(-dotProduct);
//                outputMaps[outputIndex][outputRow][outputCol] = (e_z - e_zn) / (e_z + e_zn);
            }
        }
    }
}

vector<FeatureMap> ConvolutionalLayer::singleThreadedBackPropagate(const vector<FeatureMap> &errorGradient) {
    // error check
    assert (errorGradient.size() == outputMaps.size());

    // clear the nudges desired for the previous layers
    for (auto& desired : newErrorGradient) {
        desired.clearOut();
    }

    for (size_t outputChannel = 0; outputChannel < outputMaps.size(); outputChannel++) {
        for (size_t outputRow = 0; outputRow < outputHeight; outputRow++) {
            for (size_t outputCol = 0; outputCol < outputWidth; outputCol++) {
                auto dh = errorGradient[outputChannel][outputRow][outputCol];

                for (size_t inputChannel = 0; inputChannel < inputMaps.size(); inputChannel++) {
                    for (size_t filterRow = 0; filterRow < filterHeight; filterRow++) {
                        for (size_t filterCol = 0; filterCol < filterWidth; filterCol++) {
                            // add a 'weighted' version of the output to the newErrorGradient
                            // dX[h:h+f, w:w+f] += W * dh(h,w)
                            newErrorGradient[inputChannel][filterRow + outputRow][filterCol + outputCol] += filters[outputChannel][inputChannel][filterRow][filterCol] * dh;
                            // get the update for the filters with the difference a "conv" of that error
                            // dW += X[h:h+f, w:w+f] * dH(h,w)
                            filterAdjustments[outputChannel][inputChannel][filterRow][filterCol] += inputMaps[inputChannel][filterRow + outputRow][filterCol + outputCol] * dh;
                        }
                    }
                }

            }
        }
    }

    if (parent != nullptr) {
        // be recursive
        return parent->backPropagate(newErrorGradient);
    }
    return newErrorGradient;
}

vector<FeatureMap> ConvolutionalLayer::multiThreadedBackPropagation(const vector<FeatureMap> &errorGradient) {
    // error check
    assert (errorGradient.size() == outputMaps.size());
    thread workers[inputMaps.size()];

    for (int i = 0; i < inputMaps.size(); i++) {
        workers[i] = thread(&ConvolutionalLayer::m_threading_BackPropagation, this, i, errorGradient);
    }
    for (auto& w : workers) {
        w.join();
    }

    if (parent != nullptr) {
        // be recursive
        return parent->backPropagate(newErrorGradient);
    }
    return newErrorGradient;
}

void ConvolutionalLayer::m_threading_BackPropagation(int inputMapIndex, const vector<FeatureMap> &errorGradient) {
    // clear the nudges desired for the previous layers
    newErrorGradient[inputMapIndex].clearOut();

    for (size_t outputChannel = 0; outputChannel < outputMaps.size(); outputChannel++) {
        for (size_t outputRow = 0; outputRow < outputHeight; outputRow++) {
            for (size_t outputCol = 0; outputCol < outputWidth; outputCol++) {
                auto dh = errorGradient[outputChannel][outputRow][outputCol];

                for (size_t filterRow = 0; filterRow < filterHeight; filterRow++) {
                    for (size_t filterCol = 0; filterCol < filterWidth; filterCol++) {
                        // add a 'weighted' version of the output to the newErrorGradient
                        // dX[h:h+f, w:w+f] += W * dh(h,w)
                        newErrorGradient[inputMapIndex][filterRow + outputRow][filterCol + outputCol] += filters[outputChannel][inputMapIndex][filterRow][filterCol] * dh;
                        // get the update for the filters with the difference a "conv" of that error
                        // dW += X[h:h+f, w:w+f] * dH(h,w)
                        filterAdjustments[outputChannel][inputMapIndex][filterRow][filterCol] += inputMaps[inputMapIndex][filterRow + outputRow][filterCol + outputCol] * dh;
                    }
                }
            }
        }
    }
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
        cout << "filter filterDepth : " << ch << endl;
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
        filterVelocities[i].velocityUpdateWithWeights(filterAdjustments[i]);
        //filterVelocities[i] = filterVelocities[i].velocityUpdateWithWeights() + 0.1*filterAdjustments[i];
        filters[i] = filters[i] + filterVelocities[i];
        filterAdjustments[i].clearOut();
    }

    if (parent != nullptr) {
        parent->updateError();
    }
}