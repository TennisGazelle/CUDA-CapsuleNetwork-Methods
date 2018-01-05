//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_POOLINGLAYER_H
#define NEURALNETS_POOLINGLAYER_H

#include "ICNLayer.h"

class PoolingLayer : public ICNLayer {
public:
    void calculateOutput();
private:
    size_t windowSize, strideSize;
    enum PoolingType {
        MEAN,
        MAX
    } poolingType;
};


#endif //NEURALNETS_POOLINGLAYER_H
