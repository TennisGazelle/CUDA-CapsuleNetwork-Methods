//
// Created by daniellopez on 2/22/18.
//

#include "models/VectorMap.h"

VectorMap::VectorMap(size_t height, size_t width) {
    resize(height);
    for (auto& row : (*this)) {
        row.resize(width);
    }
}

vector<VectorMap> VectorMap::toVectorMap(size_t vectorLengths, const vector<FeatureMap> inputMaps) {
    // assuming that the depth of these maps is divisible of the resulting depth (it should be)
    size_t height = inputMaps[0].size();
    size_t width = inputMaps[0][0].size();
    size_t depth = inputMaps.size()/vectorLengths;

    vector<VectorMap> result(depth, VectorMap(height, width));
    for (size_t r = 0; r < height; r++) {
        for (size_t c = 0; c < width; c++) {
            // make a vector for every "depth" vectors
            arma::vec v(vectorLengths);
            size_t resultDepth = 0;
            for (int d = 0; d < inputMaps.size(); d++) {
                if (d > 0 && d%depth == 0) {
                    result[resultDepth][r][c] = v;
                }
                v[d%depth] = inputMaps[d][r][c];
            }
        }
    }
    return result;
}

