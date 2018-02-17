//
// Created by Daniel Lopez on 1/8/18.
//

#include <cassert>
#include <Utils.h>
#include "models/Filter.h"

void Filter::init(size_t depth, size_t height, size_t width) {
    resize(depth);
    for (auto& channel : (*this)) {
        channel.resize(height);
        for (auto& row : channel) {
            row.resize(width);
            for (auto& col : row) {
                col = Utils::getRandBetween(-1, 1);
            }
        }
    }
}

void Filter::clearOut() {
    for (auto& channel : (*this)) {
        for (auto& row : channel) {
            for (auto& col : row) {
                col = 0;
            }
        }
    }
}

Filter Filter::operator+(const Filter &right) {
    assert (size() == right.size());
    assert (at(0).size() == right.at(0).size());
    assert (at(0).at(0).size() == right.at(0).at(0).size());

    Filter result = right;
    for (size_t ch = 0; ch < size(); ch++) {
        for (size_t r = 0; r < at(ch).size(); r++) {
            for (size_t c = 0; c < at(ch).at(r).size(); c++) {
                result[ch][r][c] += at(ch).at(r).at(c);
            }
        }
    }

    return result;
}