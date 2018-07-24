//
// Created by daniellopez on 6/4/18.
//

#include "GA/Individual.h"
#include <iostream>
#include <Utils.h>
#include <CapsuleNetwork/CapsuleNetwork.h>
#include <CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h>
#include <GA/CapsNetDAO.h>

Individual::Individual(int bitstringSize, const string& chromosome)
        : loss_100 (0.0), loss_300 (0.0), 
        accuracy_100 (0.0), accuracy_300 (0.0) {
    resize(bitstringSize);
    if (chromosome.empty()) {
    	generateRandom();
    	decodeChromosome();
    } else if (chromosome.size() == bitstringSize) {
        for (int i = 0; i < bitstringSize; i++) {
        	(*this)[i] = (chromosome[i] == '1');
        }
        decodeChromosome();
    } else {
        cerr << "Individual Constructor - not expected sizes" << endl;
	    exit(1);
    }
}

Individual::Individual(const Individual& src) {
    resize(src.size());
    for (int i = 0; i < src.size(); i++) {
        (*this)[i] = src.at(i);
    }
    decodeChromosome();
    accuracy_100 = src.accuracy_100;
    accuracy_300 = src.accuracy_300;
    loss_100 = src.loss_100;
    loss_300 = src.loss_300;
    crowdingDistance = src.crowdingDistance;
    rank = src.rank;
    numDominateMe = src.numDominateMe;
}

void Individual::print() const {
	cout << to_string() << endl;
}

void Individual::fullPrint() const {
    auto chromosome = to_string();
    chromosome.insert(5, 1, ' ');
    chromosome.insert(5+5+1, 1, ' ');
    chromosome.insert(5+5+5+1+1, 1, ' ');
    chromosome.insert(5+5+5+5+1+1+1, 1, ' ');
    chromosome.insert(5+5+5+5+5+1+1+1+1, 1, ' ');
    chromosome.insert(5+5+5+5+5+5+1+1+1+1+1, 1, ' ');
	cout << chromosome << ": " << endl;
	cout << "         cnInnerDim: " << capsNetConfig.cnInnerDim << endl;
	cout << "         cnOuterDim: " << capsNetConfig.cnOuterDim << endl;
	cout << "cnNumTensorChannels: " << capsNetConfig.cnNumTensorChannels << endl;
	cout << "          batchSize: " << capsNetConfig.batchSize << endl;
	cout << "             m_plus: " << capsNetConfig.m_plus << endl;
	cout << "            m_minus: " << capsNetConfig.m_minus << endl;
	cout << "             lambda: " << capsNetConfig.lambda << endl;
    cout << endl;
    cout << "     Accuracy (100): " << accuracy_100 << endl;
    cout << "         Loss (100): " << loss_100 << endl;
    cout << "     Accuracy (300): " << accuracy_300 << endl;
    cout << "          Loss(300): " << loss_300 << endl;
}

string Individual::to_string() const {
	string result = "";
	for (int i = 0; i < size(); i++) {
		result += at(i) ? '1' : '0';
	}
	return result;
}

void Individual::generateRandom() {
	for (int i = 0; i < size(); i++) {
		(*this)[i] = Utils::randomWithProbability(0.5);
	}
}

bool Individual::operator==(const Individual &other) const {
	for (unsigned int i = 0; i < size(); i++) {
		if (at(i) != other[i]) {
			return false;
		}
	}
	return (size() == other.size());
}

Individual& Individual::operator=(const Individual &other) {
    if (&other == this) {
    	return *this;
    }
    clear();
    resize(other.size());
    for (int i = 0; i < other.size(); i++) {
        (*this)[i] = other.at(i);
    }
    decodeChromosome();
    accuracy_100 = other.accuracy_100;
    accuracy_300 = other.accuracy_300;
    loss_100 = other.loss_100;
    loss_300 = other.loss_300;
    crowdingDistance = other.crowdingDistance;
    rank = other.rank;
    numDominateMe = other.numDominateMe;
    return *this;
}

void Individual::decodeChromosome() {
    auto iter = begin();
    capsNetConfig.cnInnerDim = Utils::getBinaryAsInt(vector<bool>(iter, iter+5)) + 2;
    iter += 5;
    capsNetConfig.cnOuterDim = Utils::getBinaryAsInt(vector<bool>(iter, iter+5)) + 2;
    iter += 5;
    capsNetConfig.cnNumTensorChannels = Utils::getBinaryAsInt(vector<bool>(iter, iter+5)) + 1;
    iter += 5;
    capsNetConfig.batchSize = (Utils::getBinaryAsInt(vector<bool>(iter, iter+5)) + 1) * 20;
    iter += 5;
    capsNetConfig.m_plus = double(Utils::getBinaryAsInt(vector<bool>(iter, iter+5)))/(160.0) + 0.8; 
    // div by 32, mult by 0.1, add 0.8 for range [.8, 1.0)
    iter += 5;
    capsNetConfig.m_minus = double(Utils::getBinaryAsInt(vector<bool>(iter, iter+5)))/(160.0) + 0.00625; 
    // small value for non-zero for range (0, 0.2]
    iter += 5;
    capsNetConfig.lambda = double(Utils::getBinaryAsInt(vector<bool>(iter, iter+5)))/(160.0) + 0.4; 
    //[0.4, 0.6)
}

void Individual::evaluate() {
    decodeChromosome();

    if (CapsNetDAO::getInstance()->isInDatabase(*this)) {
        CapsNetDAO::getInstance()->getFromDatabase(*this);
    } else {
        constructNetworkAndEvaluate();
        CapsNetDAO::getInstance()->addToDatabase(*this);
    }
}

void Individual::constructNetworkAndEvaluate() {
    static mutex mtx;
    mtx.lock();
    cout << "running this..." << to_string() << endl;
    fullPrint();
    CUCapsuleNetwork network(capsNetConfig);

    auto fitness = network.train();
    accuracy_100 = fitness.first;
    loss_100 = fitness.second;

//    network.train();
    fitness = network.train();
    accuracy_300 = fitness.first;
    loss_300 = fitness.second;
    mtx.unlock();

//    loss_100 = capsNetConfig.cnInnerDim * capsNetConfig.cnOuterDim * capsNetConfig.cnNumTensorChannels;
//    loss_100 += capsNetConfig.lambda * capsNetConfig.m_minus * capsNetConfig.m_plus;
////    loss_300 = loss_100*capsNetConfig.lambda;
//
//    accuracy_100 = capsNetConfig.lambda * capsNetConfig.m_plus / (capsNetConfig.cnInnerDim * capsNetConfig.cnOuterDim * capsNetConfig.cnNumTensorChannels);
////    accuracy_300 = accuracy_100*capsNetConfig.cnNumTensorChannels;
}

bool Individual::paredoDominates(const Individual &opponent) const {
    return (accuracy_100 >= opponent.accuracy_100) &&
        (accuracy_300 >= opponent.accuracy_300) &&
        (loss_100 <= opponent.loss_100) &&
        (loss_300 <= opponent.loss_300);
}

bool Individual::crowdingOperator(const Individual &opponent) const {
    if (this == &opponent) {
        return true;
    }
    return (rank < opponent.rank) ||
            ((rank == opponent.rank) && (crowdingDistance < opponent.crowdingDistance));
}

void Individual::crossoverWith(Individual &other) {
    int crossoverPoint = Utils::getRandBetween(0, size());
    for (int i = crossoverPoint; i < size(); i++) {
        iter_swap(begin()+i, other.begin()+i);
    }
    decodeChromosome();
    other.decodeChromosome();
}

void Individual::mutate() {
    int mutationPoint = Utils::getRandBetween(0, size());
    (*this)[mutationPoint] = !at(mutationPoint);
    decodeChromosome();
}

