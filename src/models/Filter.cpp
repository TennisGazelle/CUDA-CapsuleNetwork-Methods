//
// Created by Daniel Lopez on 1/8/18.
//

#include <cassert>
#include "models/Filter.h"

void Filter::clearOut() {
    for (auto& row : (*this)) {
        for (auto& col : row) {
            col = 0;
        }
    }
}

Filter Filter::operator+(const Filter &right) {
    assert (size() == right.size());
    assert (at(0).size() == right.at(0).size());

    Filter result = right;
    for (size_t r = 0; r < size(); r++) {
        for (size_t c = 0; c < at(r).size(); c++) {
            result[r][c] += at(r).at(c);
        }
    }

    return result;
}