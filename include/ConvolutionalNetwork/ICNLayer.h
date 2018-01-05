//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_ICNLAYER_H
#define NEURALNETS_ICNLAYER_H

#include <ILayer.h>
#include <models/FeatureMap.h>

class ICNLayer {
public:
    virtual void calculateOutput() = 0;

    // this class should be similar to ILayer but not have the exact same things...
    vector<FeatureMap> inputMaps, outputMaps;
};


#endif //NEURALNETS_ICNLAYER_H
