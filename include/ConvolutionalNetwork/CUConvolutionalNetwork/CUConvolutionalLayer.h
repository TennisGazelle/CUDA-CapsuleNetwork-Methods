//
// Created by daniellopez on 4/23/18.
//

#ifndef NEURALNETS_CUCONVOLUTIONALLAYER_H
#define NEURALNETS_CUCONVOLUTIONALLAYER_H


#include <models/CUUnifiedBlob.h>
#include <vector>

class CUConvolutionalLayer {
public:
    CUConvolutionalLayer(int iHeight, int iWidth, int numFilters, int fHeight, int fWidth);
    void setInput(std::vector<double> inputImage);
    void calculateOutput();

private:
    int inputHeight = 0, inputWidth = 0;
    int outputHeight = 0, outputWidth = 0;
    int filterDepth, filterHeight, filterWidth;

    CUUnifiedBlob input;
    std::vector<CUUnifiedBlob> filters;
};


#endif //NEURALNETS_CUCONVOLUTIONALLAYER_H
