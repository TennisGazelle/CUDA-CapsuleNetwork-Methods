//
// Created by daniellopez on 6/4/18.
//

#ifndef NEURALNETS_GACONFIG_H
#define NEURALNETS_GACONFIG_H

struct GAConfig {
    unsigned int bitstringSize = 5+5+6+5+5+5+5;
    unsigned int populationSize = 100;
    unsigned int numIterations = 200;

    double prob_mutation = 0.001;
    double prob_crossover = 0.5;
};

#endif //NEURALNETS_GACONFIG_H
