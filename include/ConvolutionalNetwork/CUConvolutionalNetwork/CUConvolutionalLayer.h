//
// Created by daniellopez on 4/23/18.
//

#ifndef NEURALNETS_CUCONVOLUTIONALLAYER_H
#define NEURALNETS_CUCONVOLUTIONALLAYER_H


#include <models/CUUnifiedBlob.h>
#include <vector>
#include <CapsNetConfig.h>

class CUConvolutionalLayer {
public:
    CUConvolutionalLayer(const CapsNetConfig& incomingConfig, int iHeight, int iWidth, int numFilters, int fHeight, int fWidth);
    void setInput(const std::vector<double>& inputImage);
    void forwardPropagate();
    void squashAndRemapToU(CUUnifiedBlob &u);
    void remapErrorToOutput(CUUnifiedBlob &delta_u);
    void backpropagate();
    void updateError();

private:
    int inputHeight = 0, inputWidth = 0;
    int outputHeight = 0, outputWidth = 0;
    int filterDepth, filterHeight, filterWidth, numFilters;

    CUUnifiedBlob input, delta_input, filter, filter_error, filter_velocities, output;
    CapsNetConfig config;
};


#endif //NEURALNETS_CUCONVOLUTIONALLAYER_H
