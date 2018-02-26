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

vector<arma::vec> VectorMap::toSquishedArrayOfVecs(size_t vectorDim, vector<FeatureMap> inputMaps) {
    // assuming that the depth of these maps is divisible of the resulting depth (it should be)
    size_t height = inputMaps[0].size();
    size_t width = inputMaps[0][0].size();
    size_t totalLength = height*width*inputMaps.size()/vectorDim;

    vector<arma::vec> output;
    output.reserve(totalLength);
    for (size_t r = 0; r < height; r++) {
        for (size_t c = 0; c < width; c++) {
            // make a vector for every "depth" vectors
            arma::vec v(vectorDim);
            // go down the depth of the input at this position
            for (int d = 0; d < inputMaps.size(); d++) {
                if (d%vectorDim == vectorDim-1) {
                    // squash and save as I go...
                    output.push_back(Utils::squish(v));
                }
                v[d%vectorDim] = inputMaps[d][r][c];
            }
        }
    }

    return output;
}

vector<FeatureMap> VectorMap::toArrayOfFeatureMaps(size_t desiredHeight, size_t desiredWidth, size_t desiredDepth,
                                                   const vector<arma::vec> &arrayOfVecs) {
    // assuming that desired depth is a factor of the list of Arrays

    // assuming all vectors are of the same size
    size_t vectorDim = arrayOfVecs[0].size();

    vector<FeatureMap> output(desiredDepth);
    for (auto& map : output) {
        map.setSize(desiredHeight, desiredWidth);
    }

    size_t vectorCounter = 0;
    for (size_t r = 0; r < desiredHeight; r++) {
        for (size_t c = 0; c < desiredWidth; c++) {
            for (int d = 0; d < desiredDepth; d++) {
                output[d][r][c] = arrayOfVecs[vectorCounter][d%vectorDim];
                if (d%vectorDim == vectorDim - 1) {
                    vectorCounter++;
                }
            }
        }
    }
    return output;
}