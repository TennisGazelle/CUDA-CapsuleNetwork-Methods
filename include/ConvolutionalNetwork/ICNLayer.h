//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_ICNLAYER_H
#define NEURALNETS_ICNLAYER_H

#include <ILayer.h>
#include <models/FeatureMap.h>

class ICNLayer {
public:
    virtual void init() = 0;
    void setInput(const vector<FeatureMap>& input);
    void setParentLayer(ICNLayer* parentLayer);
    void setInputDimension(size_t numInputs, size_t height, size_t width);
    void setOutputDimension(size_t numOutputs, size_t height, size_t width);
    size_t getOutputSize1D() const;
    vector<double> getOutputAsOneDimensional() const;

    virtual void calculateOutput() = 0;
    void collectInput();
    void process();

    // this class should be similar to ILayer but not have the exact same things...
    vector<FeatureMap> inputMaps, outputMaps;
    size_t inputHeight = 0, inputWidth = 0;
    size_t outputHeight = 0, outputWidth = 0;
    ICNLayer* parent = nullptr;
};


#endif //NEURALNETS_ICNLAYER_H
