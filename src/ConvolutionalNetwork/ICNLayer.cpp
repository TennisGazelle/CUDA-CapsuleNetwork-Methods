//
// Created by Daniel Lopez on 1/4/18.
//

#include <cassert>
#include "ConvolutionalNetwork/ICNLayer.h"

void ICNLayer::setParentLayer(ICNLayer *parentLayer) {
    // error check
    assert (parentLayer != nullptr);
    parent = parentLayer;
}

void ICNLayer::setInputDimension(size_t height, size_t width) {
    inputHeight = height;
    inputWidth = width;
}

void ICNLayer::setOutputDimension(size_t height, size_t width) {
    outputHeight = height;
    outputWidth = width;
}

size_t ICNLayer::getOutputSize1D() const {
    return outputHeight * outputWidth;
}