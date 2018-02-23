//
// Created by daniellopez on 2/22/18.
//

#ifndef NEURALNETS_VECTORMAP_H
#define NEURALNETS_VECTORMAP_H

#include <vector>
#include <armadillo>
#include "FeatureMap.h"

using namespace std;

class VectorMap : public vector< vector<arma::vec> > {
public:
    VectorMap(size_t height, size_t width);
    // from a set of feature maps
    static vector<VectorMap> toSquishedVectorMap(size_t vectorLength, const vector<FeatureMap> inputMaps);
    static vector<arma::vec> toSquishedArrayOfVecs(size_t vectorLength, const vector<FeatureMap> inputMaps);
};


#endif //NEURALNETS_VECTORMAP_H
