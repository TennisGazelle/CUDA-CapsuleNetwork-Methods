//
// Created by daniellopez on 6/4/18.
//

#ifndef NEURALNETS_POPULATION_H
#define NEURALNETS_POPULATION_H

#include <models/PopulationStats.h>
#include "Individual.h"

class ParedoFront;

class Population : public vector<Individual> {
public:
    Population() = default;
    void generate(int n, int bitstringSize);

    void print();
    void fullPrint();
    void getStatsFromIndividuals();
    void insertParedoFront(ParedoFront front);
    void evaluate();

    Individual tournamentSelect(bool useCrowdingOperator);
    Individual getBestIndividual() const;

    PopulationStats accuracy100;
    PopulationStats accuracy300;
    PopulationStats loss100;
    PopulationStats loss300;

    unsigned int bestIndividualIndex;
};

class ParedoFront : public vector<Individual*> {
public:
    static ParedoFront referToAsFront(Population &p);

    void assignCrowdingDistance();
    void sortByCrowdingOperator();

    void sortByAccuracy100();
    void sortByAccuracy300();
    void sortByLoss100();
    void sortByLoss300();
    void sortByChromosome();

private:
    bool isSorted = false;
};

// helper function prototypes
vector<ParedoFront> sortFastNonDominated(Population &p);


#endif //NEURALNETS_POPULATION_H
