//
// Created by daniellopez on 6/4/18.
//

#ifndef NEURALNETS_POPULATION_H
#define NEURALNETS_POPULATION_H


#include "Individual.h"

class Population : public vector<Individual> {
public:
    Population(GAConfig incomingConfig);

    void print() const;
    void fullPrint() const;
    void getStatsFromIndividuals(bool useParedoToCompare = false);
    void evaluate();

    void sortByAccuracy();
    void sortByLoss();

    Individual tournamentSelect();
    Individual getBestIndividual() const;

    double minAccuracy, maxAccuracy, averageAccuracy;
    double minLoss, maxLoss, averageLoss;

    unsigned int bestIndividualIndex, worstIndividualIndex;

private:
    GAConfig gaConfig;
    void generate(int n);
};

class ParedoFront : public Population {
public:
    ParedoFront(GAConfig incomingConfig);
};

#endif //NEURALNETS_POPULATION_H
