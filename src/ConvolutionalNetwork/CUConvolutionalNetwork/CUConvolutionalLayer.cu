//
// Created by daniellopez on 4/23/18.
//

#include <cassert>
#include <CapsNetConfig.h>
#include <CUDAUtils.h>
#include "ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.h"

CUConvolutionalLayer::CUConvolutionalLayer(const CapsNetConfig& incomingConfig, int iHeight, int iWidth, int nFilters, int fHeight, int fWidth)
        : config(incomingConfig) {
    numFilters = nFilters;
    filterDepth = 1;
    filterHeight = fHeight;
    filterWidth = fWidth;

    inputWidth = iWidth;
    inputHeight = iHeight;

    outputHeight = inputHeight - filterHeight;
    outputWidth = inputWidth - filterWidth;

    input.resize(inputHeight*inputWidth*filterDepth);
    delta_input.resize(inputHeight*inputWidth*filterDepth);
    
    filter.resize(numFilters*filterDepth*filterHeight*filterWidth);
    filter_error.resize(numFilters*filterDepth*filterHeight*filterWidth);
    filter_velocities.resize(numFilters*filterDepth*filterHeight*filterWidth);

    output.resize(outputHeight*outputWidth*numFilters);

    filter.fillWithRandom();
}

void CUConvolutionalLayer::setInput(const std::vector<double>& inputImage) {
    assert(inputImage.size() == input.getSize());
    for(int i = 0; i < input.getSize(); i++) {
        input.setValueAt_1D(i, inputImage[i]/256.0);
    }
}

void CUConvolutionalLayer::forwardPropagate() {
//    input.print("input as a feature map", inputWidth);
    CUUnifiedBlob::CUDA_convolutionalDotProduct(input, filter, output, inputHeight, inputWidth, filterHeight, filterWidth, filterDepth, numFilters);
//    filter.print("filter", filterWidth);
//    output.print("conv. layer output", outputWidth);
//    input.print("original image input", inputWidth);
}

void CUConvolutionalLayer::squashAndRemapToU(CUUnifiedBlob &u) {
//    output.print("conv. layer output", outputWidth);
    CUUnifiedBlob::CUDA_tensorFlatteningAndActivatedRemapping(u, output, outputHeight, outputWidth, config.cnNumTensorChannels, config.numClasses, config.cnInnerDim);
    output.CUDA_clear();
}

void CUConvolutionalLayer::remapErrorToOutput(CUUnifiedBlob &delta_u) {
    CUUnifiedBlob::CUDA_reconstructingTensorFromError(output, delta_u, outputHeight, outputWidth, filterDepth, config.numClasses, config.cnInnerDim);
    CUDAUtils::checkForError("after tensor reconstruction");
    output.print("cvlayer error output", outputWidth);
}

void CUConvolutionalLayer::backpropagate() {
    CUUnifiedBlob::CUDA_convolutionalBackPropFromError(output, filter, filter_error, input, delta_input, inputHeight, inputWidth, filterHeight, filterWidth, filterDepth, numFilters);
//    filter.print("filter", filterWidth);
//    filter_error.print("delta filter", filterWidth);
}

void CUConvolutionalLayer::updateError() {
    CUUnifiedBlob::elementWiseErrorUpdate(filter, filter_error, filter_velocities, filter.getSize());
}