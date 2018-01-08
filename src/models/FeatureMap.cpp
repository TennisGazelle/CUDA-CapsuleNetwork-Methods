//
// Created by Daniel Lopez on 1/4/18.
//

#include "models/FeatureMap.h"

vector<double> FeatureMap::toOneDim() const {
    vector<double> output(size() * at(0).size());

    size_t row_size = at(0).size();
    for (unsigned int r = 0; r < size(); r++) {
        for (unsigned int c = 0; c < at(r).size(); c++) {
            output[r*row_size + c] = at(r).at(c);
        }
    }

    return output;
}

FeatureMap& FeatureMap::toFeatureMap(size_t h, size_t w) {
    resize(h);
    for (auto& row : (*this)) {
        row.resize(w);
    }
    return *this;
}

void FeatureMap::setSize(size_t h, size_t w) {
    resize(h);
    for (auto& row : (*this)) {
        row.resize(w);
    }
}