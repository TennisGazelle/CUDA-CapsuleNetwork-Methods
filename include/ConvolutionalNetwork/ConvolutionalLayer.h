//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_CONVOLUTIONALLAYER_H
#define NEURALNETS_CONVOLUTIONALLAYER_H

#include "ICNLayer.h"

typedef vector <vector <double> > Filter;

class ConvolutionalLayer : public ICNLayer {
public:
    ConvolutionalLayer() = default;
    void calculateOutput(); // don't forget to perform ReLu at the end of it
private:
    // filters (constructor should include how many of them to have)
    vector<Filter> filters; // Note: THESE ARE THE WEIGHTS TO UPDATE
    size_t filterHeight, filterWidth;
};


#endif //NEURALNETS_CONVOLUTIONALLAYER_H
