//
// Created by daniellopez on 4/23/18.
//

#include <cassert>
#include <Config.h>
#include "ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.h"

CUConvolutionalLayer::CUConvolutionalLayer(int iHeight, int iWidth, int nFilters, int fHeight, int fWidth) {
    numFilters = nFilters;
    depth = 1;
    filterHeight = fHeight;
    filterWidth = fWidth;

    inputWidth = iWidth;
    inputHeight = iHeight;

    int outputHeight = inputHeight - filterHeight + 1;
    int outputWidth = inputWidth - filterWidth + 1;

    input.resize(inputHeight*inputWidth*depth);
    filter.resize(numFilters*depth*filterHeight*filterWidth);
    output.resize(outputHeight*outputWidth*numFilters);

    filter.fillWithRandom();
}

void CUConvolutionalLayer::setInput(std::vector<double> inputImage) {
    assert(inputImage.size() == input.getSize());
    for(int i = 0; i < input.getSize(); i++) {
        input.setValueAt_1D(i, inputImage[i]);
    }
}

void CUConvolutionalLayer::calculateOutput() {
    CUUnifiedBlob::CUDA_convolutionalDotProduct(input, filter, output, inputHeight, inputWidth, filterHeight, filterWidth, depth, numFilters);
}

void CUConvolutionalLayer::squashAndRemapToU(CUUnifiedBlob &u) {
    CUUnifiedBlob::CUDA_tensorFlatteningAndActivatedRemapping(u, output, outputHeight, outputWidth, numFilters/Config::cnInnerDim, Config::numClasses, Config::cnInnerDim);
}