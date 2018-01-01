//
// Created by Daniel Lopez on 12/28/17.
//

#include "ILayer.h"

//void ILayer::setInputSize(size_t pInputSize) {
//    inputSize = pInputSize;
//}
//
//void ILayer::setOutputSize(size_t pOutputSize) {
//    outputSize = pOutputSize;
//}

void ILayer::setInput(const vector<double> pInput) {
    input = pInput;
}

vector<double> const& ILayer::getOutput() const {
    return output;
}