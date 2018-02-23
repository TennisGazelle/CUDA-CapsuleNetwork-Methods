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

    // from a set of feature maps
    static vector<VectorMap> toVectorMap(size_t resultingDepth, const vector<FeatureMap> inputMaps);
};


#endif //NEURALNETS_VECTORMAP_H
