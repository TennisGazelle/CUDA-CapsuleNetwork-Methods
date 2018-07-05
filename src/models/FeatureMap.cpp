//
// Created by Daniel Lopez on 1/4/18.
//

#include <iostream>
#include <iomanip>
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

vector<FeatureMap> FeatureMap::toFeatureMaps(size_t h, size_t w, const vector<double> &cubeData) {
    size_t depth = cubeData.size() / (h * w);
    vector<FeatureMap> maps(depth);
    for (auto& map : maps) {
        map.setSize(h, w);
    }

    int cubeIndex = 0;
    for (size_t mapIndex = 0; mapIndex < depth; mapIndex++) {
        for (size_t r = 0; r < h; r++) {
            for (size_t c = 0; c < w; c++) {
                maps[mapIndex][r][c] = cubeData[cubeIndex++];
            }
        }
    }

    return maps;
}

void FeatureMap::setSize(size_t h, size_t w) {
    resize(h);
    for (auto& row : (*this)) {
        row.resize(w);
    }
}

void FeatureMap::clearOut() {
    for (auto& row : (*this)) {
        for (auto& col : row) {
            col = 0;
        }
    }
}

void FeatureMap::print() const {
//    cout << fixed << setprecision(7);
    for (auto row : (*this)) {
        for (auto col : row) {
            cout << col << "\t";
        }
        cout << endl;
    }
}