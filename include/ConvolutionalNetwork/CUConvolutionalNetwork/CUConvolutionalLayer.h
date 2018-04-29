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
    void squashAndRemapToU(CUUnifiedBlob &u);

private:
    int inputHeight = 0, inputWidth = 0;
    int outputHeight = 0, outputWidth = 0;
    int depth, filterHeight, filterWidth, numFilters;

    CUUnifiedBlob input, filter, output;
//    std::vector<CUUnifiedBlob> filters;
//    CUUnifiedBlob filter;
};


#endif //NEURALNETS_CUCONVOLUTIONALLAYER_H