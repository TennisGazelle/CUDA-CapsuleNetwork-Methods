//
// Created by daniellopez on 2/22/18.
//

#include <Utils.h>
#include "models/VectorMap.h"

VectorMap::VectorMap(size_t height, size_t width) {
    resize(height);
    for (auto& row : (*this)) {
        row.resize(width);
    }
}

vector<VectorMap> VectorMap::toSquishedVectorMap(size_t vectorLength, const vector<FeatureMap> inputMaps) {
    // assuming that the depth of these maps is divisible of the resulting depth (it should be)
    size_t height = inputMaps[0].size();
    size_t width = inputMaps[0][0].size();
    size_t depth = inputMaps.size()/vectorLength;

    vector<VectorMap> output(depth, VectorMap(height, width));
    for (size_t r = 0; r < height; r++) {
        for (size_t c = 0; c < width; c++) {
            // make a vector for every "depth" vectors
            arma::vec v(vectorLength);
            size_t depthInOutput = 0;
            // go down the depth of the input at this position
            for (int d = 0; d < inputMaps.size(); d++) {
                if (d%vectorLength == vectorLength-1) {
                    // squash and save
                    output[depthInOutput++][r][c] = Utils::squish(v);
                }
                v[d%vectorLength] = inputMaps[d][r][c];
            }
        }
    }
    return output;
}

vector<arma::vec> VectorMap::toSquishedArrayOfVecs(size_t vectorLength, const vector<FeatureMap> inputMaps) {
    // assuming that the depth of these maps is divisible of the resulting depth (it should be)
    size_t height = inputMaps[0].size();
    size_t width = inputMaps[0][0].size();
    size_t totalLength = height*width*inputMaps.size()/vectorLength;

    vector<arma::vec> output;
    output.reserve(totalLength);
    for (size_t r = 0; r < height; r++) {
        for (size_t c = 0; c < width; c++) {
            // make a vector for every "depth" vectors
            arma::vec v(vectorLength);
            // go down the depth of the input at this position
            for (int d = 0; d < inputMaps.size(); d++) {
                if (d%vectorLength == vectorLength-1) {
                    // squash and save as I go...
                    output.push_back(Utils::squish(v));
                }
                v[d%vectorLength] = inputMaps[d][r][c];
            }
        }
    }

    return output;
}