//
// Created by daniellopez on 6/4/18.
//

#include "GA/Population.h"
#include <iostream>
#include <cfloat>
#include <algorithm>
#include <Utils.h>
#include <assert.h>
#include <thread>
#include <book.h>

void Population::generate(int n, int bitstringSize) {
	clear();
	for (int i = 0; i < n; i++) {
		emplace_back(Individual(bitstringSize));
	}
}

void Population::print() {
    ParedoFront pointers = ParedoFront::referToAsFront(*this);
    pointers.sortByChromosome();
    string lastChromosome = pointers[0]->to_string();
    string newChromosome;

	for (int i = 0; i < size(); i++) {
        // add break between sets of duplicates
        newChromosome = pointers[i]->to_string();
        if (pointers[i]->to_string() != lastChromosome) {
//            cout << endl;
        }

        cout << "[" << (i < 10 ? "0" : "") << i << "] - " << newChromosome << endl;

        lastChromosome = newChromosome;
	}
}

void Population::fullPrint() {
    ParedoFront pointers = ParedoFront::referToAsFront(*this);
    pointers.sortByChromosome();
	for (int i = 0; i < size(); i++) {
		cout << "[" << (i < 10 ? "0" : "") << i << "] - ";
        pointers[i]->fullPrint();
		cout << endl;
	}
}

void* evalIndividual(void* data) {
    Individual* indiv = (Individual*) data;
    indiv->evaluate();
}

void Population::evaluate() {
//    static ctpl::thread_pool p(4);
//    vector< future<void> > tasks(size());
//
//    for (int i = 0; i < size(); i++) {
//        tasks[i] = p.push(evalIndividual, std::ref(at(i)));
//    }
//    for (int i = 0; i < size(); i++) {
//        tasks[i].get();
//    }

//    auto evaluees = ParedoFront::referToUniqueIndividuals(*this);
//    CUTThread threads[evaluees.first.size()];
//
//    for (int i = 0; i < evaluees.first.size(); i++) {
//        threads[i] = start_thread(evalIndividual, evaluees.first[i]);
//    }
//    for (int i = 0; i < evaluees.first.size(); i++) {
//        end_thread(threads[i]);
//        for (int j = 0; j < evaluees.second.size(); j++) {
//            if (evaluees.first[i]->to_string() == evaluees.second[j]->to_string()) {
//                (*evaluees.second[j]) = (*evaluees.first[i]);
//            }
//        }
//    }

    for (auto &indiv : (*this)) {
        indiv.evaluate();
    }
	getStatsFromIndividuals();
}

void Population::getStatsFromIndividuals() {
    accuracy100.reset();
    accuracy300.reset();
    loss100.reset();
    loss300.reset();

	for (unsigned int i = 0; i < size(); i++) {
		accuracy100.min = min(accuracy100.min, at(i).accuracy_100);
		accuracy100.max = max(accuracy100.max, at(i).accuracy_100);
		accuracy300.min = min(accuracy300.min, at(i).accuracy_300);
		accuracy300.max = max(accuracy300.max, at(i).accuracy_300);

		loss100.min = min(loss100.min, at(i).loss_100);
		loss100.max = max(loss100.max, at(i).loss_100);
		loss300.min = min(loss300.min, at(i).loss_300);
		loss300.max = max(loss300.max, at(i).loss_300);

		accuracy100.average += at(i).accuracy_100;
		accuracy300.average += at(i).accuracy_300;
		loss100.average += at(i).loss_100;
		loss300.average += at(i).loss_300;
	}

	accuracy100.average /= size();
	accuracy300.average /= size();
	loss100.average /= size();
	loss300.average /= size();
}

void Population::insertParedoFront(ParedoFront front) {
    for (Individual* individual : front) {
        push_back(*individual);
    }
}

Individual Population::tournamentSelect(bool useCrowdingOperator) {
    int left = Utils::getRandBetween(0, size());
    int right = Utils::getRandBetween(0, size());
    if (right == left){
        right = (right+1)%size();
    }

    if (useCrowdingOperator) {
        if (at(left).paredoDominates(at(right))) {
            return at(left);
        } else {
            return at(right);
        }
    } else {
        if (at(left).rank > at(right).rank) {
            return at(left);
        } else {
            return at(right);
        }
    }
}

void ParedoFront::assignCrowdingDistance() {
    if (isSorted || empty())
        return;

    isSorted = true;

    for (auto i : (*this)) {
        i->crowdingDistance = 0;
    }
    // accuracy_100
    sortByAccuracy100();
    (*this)[0]->crowdingDistance = DBL_MAX;
    (*this)[size()-1]->crowdingDistance = DBL_MAX;
    for (unsigned int i = 1; i < size()-1; i++) {
        double numerator = at(i+1)->accuracy_100 - at(i-1)->accuracy_100;
        double denominator = 100;
        (*this)[i]->crowdingDistance += numerator/denominator;
    }

    // loss_100
    sortByLoss100();
    (*this)[0]->crowdingDistance = DBL_MAX;
    (*this)[size()-1]->crowdingDistance = DBL_MAX;
    for (unsigned int i = 1; i < size()-1; i++) {
        // we have to flip these instead (because we're minimizing this number)
        double numerator = at(i-1)->loss_100 - at(i+1)->loss_100;
        double denominator = 8000.0;
        (*this)[i]->crowdingDistance += numerator/denominator;
    }

    // accuracy_300
    sortByAccuracy300();
    (*this)[0]->crowdingDistance = DBL_MAX;
    (*this)[size()-1]->crowdingDistance = DBL_MAX;
    for (unsigned int i = 1; i < size()-1; i++) {
        double numerator = at(i+1)->accuracy_300 - at(i-1)->accuracy_300;
        double denominator = 100;
        (*this)[i]->crowdingDistance += numerator/denominator;
    }

    // loss_300
    sortByLoss300();
    (*this)[0]->crowdingDistance = DBL_MAX;
    (*this)[size()-1]->crowdingDistance = DBL_MAX;
    for (unsigned int i = 1; i < size()-1; i++) {
        // we have to flip these instead (because we're minimizing this number)
        double numerator = at(i-1)->loss_300 - at(i+1)->loss_300;
        double denominator = 8000.0;
        (*this)[i]->crowdingDistance += numerator/denominator;
    }
}

void ParedoFront::sortByCrowdingOperator() {
    std::sort(begin(), end(), [](Individual* i, Individual* j) -> bool {
        return i->crowdingOperator(*j);
    });
}

vector<ParedoFront> sortFastNonDominated(Population &p) {
    vector<ParedoFront> fronts(1);

    for (unsigned int i = 0; i < p.size(); i++) {
        p[i].individualsIDominate.clear();
        p[i].numDominateMe = 0;
        p[i].rank = 2;

        for (unsigned int j = 0; j < p.size(); j++) {
            if (j == i) {
                continue;
            }

            if (p[i].paredoDominates(p[j])) {
                p[i].individualsIDominate.push_back(&p[j]);
            } else if (p[j].paredoDominates(p[i])) {
                p[i].numDominateMe++;
            }
        }

        if (p[i].numDominateMe == 0) {
            p[i].rank = 1;
            fronts[0].push_back(&p[i]);
        }
    }

    // for each person in the first front
    ParedoFront nextFront;
    int currentFrontIndex = 0;

    // while this front doesn't have people that are empty
    while (!fronts[currentFrontIndex].empty()) {
        for (auto p1 : fronts[currentFrontIndex]) {
            for (auto p2 : p1->individualsIDominate) {
                p2->numDominateMe--;
                if (p2->numDominateMe == 0) {
                    p2->rank = currentFrontIndex+2;
                    nextFront.push_back(p2);
                }
            }
        }
        currentFrontIndex++;
        fronts.push_back(nextFront);
        nextFront.clear();
    }

    return fronts;
}

void ParedoFront::sortByAccuracy100() {
    sort(begin(), end(), [](Individual* i, Individual* j) -> bool {
        return (i->accuracy_100 < j->accuracy_100);
    });
}

void ParedoFront::sortByAccuracy300() {
    sort(begin(), end(), [](Individual* i, Individual* j) -> bool {
        return (i->accuracy_300 < j->accuracy_300);
    });
}

void ParedoFront::sortByLoss100() {
    sort(begin(), end(), [](Individual* i, Individual* j) -> bool {
        return (i->loss_100 > j->loss_100);
    });
}

void ParedoFront::sortByLoss300() {
    sort(begin(), end(), [](Individual* i, Individual* j) -> bool {
        return (i->loss_300 > j->loss_300);
    });
}

bool chromosomeSortingFunc(Individual* i, Individual* j) {
    for (int index = 0; index < i->size(); index++) {
        if(i->at(index) < j->at(index)) {
            return true;
        } else if (i->at(index) > j->at(index)) {
            return false;
        }
    }
    return true;
}

void ParedoFront::sortByChromosome() {
//    sort(begin(), end(), chromosomeSortingFunc);
//    sort(begin(), end(), [](const Individual* i, const Individual* j) -> bool {
//        for (int index = 0; index < i->size(); index++) {
//            if(i->at(index) < j->at(index)) {
//                return true;
//            } else if (i->at(index) > j->at(index)) {
//                return false;
//            }
//        }
//        return true;
//    });
}

ParedoFront ParedoFront::referToAsFront(Population &p) {
    ParedoFront result;
    result.reserve(p.size());
    for (auto& i : p) {
        result.push_back(&i);
    }
    return result;
}

pair<ParedoFront, ParedoFront> ParedoFront::referToUniqueIndividuals(Population &p) {
    ParedoFront unique, duplicates;
    unique.reserve(p.size());
    for (int i = 0; i < p.size(); i++) {
        bool isUnique = true;
        for (int j = 0; j < unique.size(); j++) {
            if (p[i].to_string() == unique[j]->to_string()) {
                isUnique = false;
                break;
            }
        }

        if (isUnique) {
            unique.push_back(&p[i]);
        } else {
            duplicates.push_back(&p[i]);
        }
    }
    return {unique, duplicates};
}