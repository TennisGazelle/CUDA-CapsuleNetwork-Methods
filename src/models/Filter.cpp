//
// Created by Daniel Lopez on 1/8/18.
//

#include "models/Filter.h"

void Filter::clearOut() {
    for (auto& row : (*this)) {
        for (auto& col : row) {
            col = 0;
        }
    }
}