//
// Created by daniellopez on 6/4/18.
//

#ifndef NEURALNETS_GA_H
#define NEURALNETS_GA_H

#include "Population.h"

class GA {
public:
    GA(GAConfig incomingConfig);
    void collectStats();
    void printStats() const;

    void makeNextGen(bool useCrowdingOperator = false);
    void NSGARun();
    void printFeaturesOfBestIndividual() const;
    Population getParentPopulation() const;

private:
    Population parentPop, childPop;
    GAConfig gaConfig;
    vector<PopulationStats> accuracy100Timeline, accuracy300Timeline, loss100Timeline, loss300Timeline;
    void NSGAStep();
};

/// Pure host-testable generation mechanics. Inputs are intentionally passed by
/// value so callers cannot accidentally mutate the current generation while
/// constructing or truncating the next one.
Population makeOffspringGeneration(Population parents,
                                   const GAConfig& config,
                                   bool useCrowdingOperator = false);
Population selectNextNSGAGeneration(Population combinedPopulation,
                                    std::size_t targetSize);

#endif //NEURALNETS_GA_H
