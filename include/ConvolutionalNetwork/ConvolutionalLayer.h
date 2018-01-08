//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_CONVOLUTIONALLAYER_H
#define NEURALNETS_CONVOLUTIONALLAYER_H

#include "ICNLayer.h"

typedef vector <vector <double> > Filter;

class ConvolutionalLayer : public ICNLayer {
public:
    ConvolutionalLayer(size_t iHeight, size_t iWidth, size_t numFilters, size_t fHeight = 5, size_t fWidth = 5);
    ConvolutionalLayer(ICNLayer* pParent, size_t numFilters, size_t fHeight = 5, size_t fWidth = 5);
    void init();
    void calculateOutput();
    void backPropagate(const vector<FeatureMap>& errorGradient);
private:
    double dotMatrixWithFilter(int beginRow, int beginCol, int filterIndex) const;
    // filters (constructor should include how many of them to have)
    vector<Filter> filters; // Note: THESE ARE THE WEIGHTS TO UPDATE
    size_t filterHeight, filterWidth;
};


#endif //NEURALNETS_CONVOLUTIONALLAYER_H
