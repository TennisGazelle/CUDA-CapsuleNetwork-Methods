//
// Created by daniellopez on 7/23/18.
//

#include <models/PopulationStats.h>

#include <limits>

void PopulationStats::reset() {
    min = std::numeric_limits<double>::infinity();
    max = -std::numeric_limits<double>::infinity();
    average = 0.0;
}
