//
// Created by daniellopez on 4/23/18.
//

#include <cassert>
#include "ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.h"

CUConvolutionalLayer::CUConvolutionalLayer(int iHeight, int iWidth, int numFilters, int fHeight, int fWidth) {
    filterDepth = 1;
    filterHeight = fHeight;
    filterWidth = fWidth;

    inputWidth = iWidth;
    inputHeight = iHeight;

    filters.resize(numFilters);
    for (auto& blob : filters) {
        blob.resize(filterDepth * filterHeight * filterWidth);
        blob.fillWithRandom();
    }
}

void CUConvolutionalLayer::setInput(std::vector<double> inputImage) {
    assert(inputImage.size() == input.getSize());
    for(int i = 0; i < input.getSize(); i++) {
        input.setValueAt_1D(i, inputImage[i]);
    }
}

void CUConvolutionalLayer::calculateOutput() {

}