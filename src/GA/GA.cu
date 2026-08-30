//
// Created by daniellopez on 6/4/18.
//

#include <iostream>
#include <Utils.h>
#include "GA/GA.h"

GA::GA(GAConfig incomingConfig) : gaConfig(incomingConfig) {
    parentPop.generate(gaConfig.populationSize, gaConfig.bitstringSize);
    parentPop.evaluate();
    collectStats();
}

void GA::collectStats() {
    accuracy100Timeline.push_back(parentPop.accuracy100);
    accuracy300Timeline.push_back(parentPop.accuracy300);
    loss100Timeline.push_back(parentPop.loss100);
    loss300Timeline.push_back(parentPop.loss300);
}

void GA::printStats() const {
    cout << "MIN:\tA100\tA300\tL100\tL300" << endl;
    for (std::size_t i = 0; i < accuracy100Timeline.size(); ++i) {
        cout << "\t" << accuracy100Timeline[i].min << "\t"
             << accuracy300Timeline[i].min << "\t"
             << loss100Timeline[i].min << "\t"
             << loss300Timeline[i].min << endl;
    }

    cout << "AVERAGE:\tA100\tA300\tL100\tL300" << endl;
    for (std::size_t i = 0; i < accuracy100Timeline.size(); ++i) {
        cout << "\t" << accuracy100Timeline[i].average << "\t"
             << accuracy300Timeline[i].average << "\t"
             << loss100Timeline[i].average << "\t"
             << loss300Timeline[i].average << endl;
    }

    cout << "MAX:\tA100\tA300\tL100\tL300" << endl;
    for (std::size_t i = 0; i < accuracy100Timeline.size(); ++i) {
        cout << "\t" << accuracy100Timeline[i].max << "\t"
             << accuracy300Timeline[i].max << "\t"
             << loss100Timeline[i].max << "\t"
             << loss300Timeline[i].max << endl;
    }
}

void GA::NSGARun() {
    cout << "NSGA RUN" << endl;
    for (unsigned int i = 0; i < gaConfig.numIterations; ++i) {
        cout << "Iteration (NSGA-II): " << i << endl;
        parentPop.getStatsFromIndividuals();
        collectStats();
        parentPop.fullPrint();

        makeNextGen(i != 0);
        NSGAStep();
    }
    parentPop.fullPrint();
}

void GA::NSGAStep() {
    const std::size_t targetSize = parentPop.size();
    Population combined = parentPop;
    combined.insert(combined.end(), childPop.begin(), childPop.end());
    parentPop = selectNextNSGAGeneration(combined, targetSize);
    parentPop.getStatsFromIndividuals();
}

void GA::makeNextGen(bool useCrowdingOperator) {
    childPop = makeOffspringGeneration(parentPop, gaConfig, useCrowdingOperator);
    childPop.evaluate();
    childPop.getStatsFromIndividuals();
}

Population GA::getParentPopulation() const {
    return parentPop;
}

void GA::printFeaturesOfBestIndividual() const {
    if (parentPop.empty()) {
        cout << "No individuals available." << endl;
        return;
    }
    parentPop.getBestIndividual().fullPrint();
}
