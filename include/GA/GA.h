//
// Created by daniellopez on 6/4/18.
//

#ifndef NEURALNETS_GA_H
#define NEURALNETS_GA_H


#include "Population.h"

class GA {
public:
    GA(GAConfig incomingConfig);
    void init();
    void makeNextGen(bool useParedoToCompare = false);
    void NSGAStep();
    void NSGARun();
    void run();
    void collectStats();
    void printFeaturesOfBestIndividual() const;


    Population parentPop, childPop;
private:
    GAConfig gaConfig;
};


#endif //NEURALNETS_GA_H
