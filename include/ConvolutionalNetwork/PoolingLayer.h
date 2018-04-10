//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_POOLINGLAYER_H
#define NEURALNETS_POOLINGLAYER_H

#include <models/PoolingType.h>
#include "ICNLayer.h"

class PoolingLayer : public ICNLayer {
public:
    explicit PoolingLayer(ICNLayer* parent, PoolingType pt = MAX, size_t sSize = 2, size_t wHeight = 5, size_t wWidth = 5);
    void init();
    void calculateOutput();
    void outputLayerToFile(ofstream &fout) const;
    vector<FeatureMap> singleThreadedBackPropagate(const vector<FeatureMap>& errorGradient);
    vector<FeatureMap> multiThreadedBackPropagation(const vector<FeatureMap>& errorGradient);
    void updateError() = 0; // TODO write this eventually
private:
    double findPoolValue(size_t rowBegin, size_t colBegin, size_t channel);
    pair<size_t, size_t> returnCoordinatesOfHighest(size_t rowBegin, size_t colBegin, size_t channel);
    size_t windowHeight, windowWidth, strideSize;
    PoolingType poolingType;
};


#endif //NEURALNETS_POOLINGLAYER_H
