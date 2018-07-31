//
// Created by daniellopez on 4/23/18.
//

#include <cassert>
#include <CapsNetConfig.h>
#include <CUDAUtils.h>
#include <iostream>
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

    totalMemoryUsage += 2*(inputHeight*inputWidth*filterDepth);
    totalMemoryUsage += 3*(numFilters*filterDepth*filterHeight*filterWidth);
    totalMemoryUsage += outputHeight*outputWidth*numFilters;
}

void CUConvolutionalLayer::setInput(const std::vector<double>& inputImage) {
    assert(inputImage.size() == input.getSize());
    for (int i = 0; i < input.getSize(); i++) {
        input.setValueAt_1D(i, inputImage[i]/256.0);
    }
}

void CUConvolutionalLayer::setInput(const Image &inputImage) {
    assert(inputImage.size() == input.getSize());
    for (int i = 0; i < input.getSize(); i++) {
        input.setValueAt_1D(i, double(inputImage[i])/256.0);
    }
}

void CUConvolutionalLayer::forwardPropagate() {
//    input.print("input as a feature map", inputWidth);
    cudaDeviceSynchronize();
    CUUnifiedBlob::CUDA_convolutionalDotProduct(input, filter, output, inputHeight, inputWidth, filterHeight, filterWidth, filterDepth, numFilters);
    if (output.CUDA_hasNan()) {
        cerr << "convolutional output has a nan: " << output.hasNan() << endl;
        output.print("output", outputWidth);
        filter.print("filter", filterWidth);
        exit(1);
     }
//    if (output.isAllZeros()) {
//        cerr << "convolutional output is all zeros..." << endl;
//        CUUnifiedBlob::CUDA_convolutionalDotProduct(input, filter, output, inputHeight, inputWidth, filterHeight, filterWidth, filterDepth, numFilters);
//        printInput();
//        printFilter();
//        printOutput();
//        exit(1);
//    }
}

void CUConvolutionalLayer::squashAndRemapToU(CUUnifiedBlob &u) {
//    output.print("conv. layer output", outputWidth);
    u.CUDA_clear();
    CUUnifiedBlob::CUDA_tensorFlatteningAndActivatedRemapping(u, output, outputHeight, outputWidth, config.cnNumTensorChannels, config.numClasses, config.cnInnerDim);
//    output.CUDA_clear();
}

void CUConvolutionalLayer::remapErrorToOutput(CUUnifiedBlob &delta_u) {
    CUUnifiedBlob::CUDA_reconstructingTensorFromError(output, delta_u, outputHeight, outputWidth, config.cnNumTensorChannels, config.numClasses, config.cnInnerDim);
//    delta_u.print("delta_u", config.cnInnerDim);
//    output.print("cvlayer error output", outputWidth);
}

void CUConvolutionalLayer::backPropagate() {
    CUUnifiedBlob::CUDA_convolutionalBackPropFromError(output, filter, filter_error, input, delta_input, inputHeight, inputWidth, filterHeight, filterWidth, filterDepth, numFilters);
//    filter.print("filter", filterWidth);
//    filter_error.print("delta filter", filterWidth);
//    if (filter.hasNan() != -1) {
//        cerr << "filter in conv. layer has nan: " << filter.hasNan() << endl;
//        cerr << "dumping..." << endl;
//        output.print("cvlayer error output", outputWidth);
//        exit(1);
//    }
//    if (filter_error.hasNan() != -1) {
//        cerr << "filter_error in conv. layer has nan: " << filter_error.hasNan() << endl;
//        cerr << "dumping..." << std::endl;
//        output.print("cvlayer error output", outputWidth);
//        exit(1);
//    }
}

void CUConvolutionalLayer::updateError() {
    CUUnifiedBlob::CUDA_elementWiseErrorUpdate(filter, filter_error, filter_velocities, filter.getSize());
}

void CUConvolutionalLayer::printFilter() const {
    filter.print("filter", filterWidth);
}

void CUConvolutionalLayer::printInput() const {
    input.print("original image input", inputWidth);
}

void CUConvolutionalLayer::printOutput() const {
    output.print("output", outputWidth);
}

int CUConvolutionalLayer::getTotalMemoryUsage() const {
    return totalMemoryUsage;
}