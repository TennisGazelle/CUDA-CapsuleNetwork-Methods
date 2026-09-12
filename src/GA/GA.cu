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
    for (int i = 0; i < accuracy100Timeline.size(); i++) {
        cout << "\t";
        cout << accuracy100Timeline[i].min << "\t";
        cout << accuracy300Timeline[i].min << "\t";
        cout << loss100Timeline[i].min << "\t";
        cout << loss300Timeline[i].min << "\t";
        cout << endl;
    }

    cout << "AVERAGE:\tA100\tA300\tL100\tL300" << endl;
    for (int i = 0; i < accuracy100Timeline.size(); i++) {
        cout << "\t";
        cout << accuracy100Timeline[i].average << "\t";
        cout << accuracy300Timeline[i].average << "\t";
        cout << loss100Timeline[i].average << "\t";
        cout << loss300Timeline[i].average << "\t";
        cout << endl;
    }

    cout << "MAX:\tA100\tA300\tL100\tL300" << endl;
    for (int i = 0; i < accuracy100Timeline.size(); i++) {
        cout << "\t";
        cout << accuracy100Timeline[i].max << "\t";
        cout << accuracy300Timeline[i].max << "\t";
        cout << loss100Timeline[i].max << "\t";
        cout << loss300Timeline[i].max << "\t";
        cout << endl;
    }
}

void GA::NSGARun() {
    cout << "NSGA RUN" << endl;
    for (int i = 0; i < gaConfig.numIterations; i++) {
        cout << "Iteration (NSGA-II): " << i << endl;
        parentPop.getStatsFromIndividuals();
        collectStats();
//        parentPop.print();
        parentPop.fullPrint();

        makeNextGen(i != 0);
        NSGAStep();
    }
    parentPop.fullPrint();
}

void GA::NSGAStep() {
    unsigned long N = parentPop.size();
    childPop.insert(childPop.end(), parentPop.begin(), parentPop.end());
    vector<ParedoFront> f = sortFastNonDominated(childPop);
    parentPop.clear();
    int index = 0;

    // until the parent population is filled
    while (index < f.size() && (parentPop.size() + f[index].size()) <= N) {
        // include the i-th nondominated front to the parent pop
        parentPop.insertParedoFront(f[index]);
        index++;
    }
    if (parentPop.size() < N) {
        f[index].assignCrowdingDistance();
        f[index].sortByCrowdingOperator();
        parentPop.insertParedoFront(f[index]);
    }
    parentPop.erase(parentPop.begin()+N, parentPop.end());
}

void GA::makeNextGen(bool useCrowdingOperator) {
    childPop.clear();
    for (int i = 0; i < parentPop.size(); i++) {
        childPop.push_back(parentPop.tournamentSelect(useCrowdingOperator));

        if (Utils::randomWithProbability(gaConfig.prob_mutation)) {
            childPop[i].mutate();
        }

        if (Utils::randomWithProbability(gaConfig.prob_crossover)) {
           int crossoverIndex = Utils::getRandBetween(0, parentPop.size());
           childPop[i].crossoverWith(parentPop[crossoverIndex]);
           childPop.push_back(parentPop[crossoverIndex]);
           i++;
        }
    }
    childPop.evaluate();
    childPop.getStatsFromIndividuals();
}

Population GA::getParentPopulation() const {
    return parentPop;
}