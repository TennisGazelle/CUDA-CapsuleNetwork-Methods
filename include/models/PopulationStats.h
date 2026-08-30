//
// Created by daniellopez on 6/6/18.
//

#ifndef NEURALNETS_POPULATIONSTATS_H
#define NEURALNETS_POPULATIONSTATS_H

#include <limits>

struct PopulationStats {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double average = 0.0;

    void reset();
};

#endif //NEURALNETS_POPULATIONSTATS_H
