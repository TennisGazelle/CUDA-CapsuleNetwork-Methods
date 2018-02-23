//
// Created by daniellopez on 2/22/18.
//

#include "models/VectorMap.h"

vector<VectorMap> VectorMap::toVectorMap(size_t resultingDepth, const vector<FeatureMap> inputMaps) {
    // assuming that the depth of these maps is divisible of the resulting depth (it should be)
    size_t height = inputMaps[0].size();
    size_t width = inputMaps[0][0].size();
    size_t depth = inputMaps.size()/resultingDepth;

    vector<VectorMap> result(depth, VectorMap())
}