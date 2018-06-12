//
// Created by daniellopez on 6/4/18.
//

#include "GA/Individual.h"
#include <iostream>
#include <Utils.h>
#include <CapsuleNetwork/CapsuleNetwork.h>
#include <CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h>

Individual::Individual(int expectedSize, const string& chromosome) 
        : loss_100 (0.0), loss_300 (0.0), 
        accuracy_100 (0.0), accuracy_300 (0.0) {
    resize(expectedSize);
    if (chromosome.empty()) {
    	generateRandom();
    	decodeChromosome();
    } else if (chromosome.size() == expectedSize) {
        for (int i = 0; i < expectedSize; i++) {
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
    chromosome.insert(5+5+6+1+1, 1, ' ');
    chromosome.insert(5+5+6+5+1+1+1, 1, ' ');
    chromosome.insert(5+5+6+5+5+1+1+1+1, 1, ' ');
    chromosome.insert(5+5+6+5+5+5+1+1+1+1+1, 1, ' ');
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
    capsNetConfig.cnInnerDim = Utils::getBinaryAsInt(vector<bool>(iter, iter+5)) + 1;
    iter += 5;
    capsNetConfig.cnOuterDim = Utils::getBinaryAsInt(vector<bool>(iter, iter+5)) + 1;
    iter += 5;
    capsNetConfig.cnNumTensorChannels = Utils::getBinaryAsInt(vector<bool>(iter, iter+6)) + 1;
    iter += 6;
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
//    CapsuleNetwork network(capsNetConfig);
//    network.runEpoch();
//    auto fitness = network.tally(false);
//    accuracy_100 = fitness.first;
//    loss_100 = fitness.second;

    loss_100 = capsNetConfig.cnInnerDim * capsNetConfig.cnOuterDim * capsNetConfig.cnNumTensorChannels;
    loss_100 += capsNetConfig.lambda * capsNetConfig.m_minus * capsNetConfig.m_plus;
    loss_300 = loss_100*1.5*capsNetConfig.cnNumTensorChannels;

    accuracy_100 = capsNetConfig.lambda * capsNetConfig.m_minus * capsNetConfig.m_plus / (capsNetConfig.cnInnerDim * capsNetConfig.cnOuterDim * capsNetConfig.cnNumTensorChannels);
    accuracy_300 = accuracy_100*1.5*capsNetConfig.cnNumTensorChannels;
}

bool Individual::paredoDominates(const Individual &opponent) const {
    return (accuracy_100 >= opponent.accuracy_100) &&
        (accuracy_300 >= opponent.accuracy_300) &&
        (loss_100 >= opponent.loss_100) &&
        (loss_300 >= opponent.loss_300);
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
}

void Individual::mutate() {
    int mutationPoint = Utils::getRandBetween(0, size());
    (*this)[mutationPoint] = !at(mutationPoint);
}

CapsuleNetworkDAO::CapsuleNetworkDAO() {
    
}

void CapsuleNetworkDAO::run() {
    connection c("dbname=cs_776 user=system password=SYSTEM host=hpcvis3.cse.unr.edu");
    work txn(c);
    result rows = txn.exec("select * from tss_dev.users_features where classification = 2 limit 10;");

    for (auto row : rows) {
        cout << row[0].c_str() << endl;
    }
//    for (auto row : rows) {
//        cout << "DATABASE ROW: " << row[0].c_str() << endl;
//    }
}

