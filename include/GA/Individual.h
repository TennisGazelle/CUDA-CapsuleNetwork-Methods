//
// Created by daniellopez on 6/4/18.
//

#ifndef NEURALNETS_INDIVIDUAL_H
#define NEURALNETS_INDIVIDUAL_H

#include <vector>
#include <string>
#include <pqxx/pqxx>
#include <CapsNetConfig.h>
#include <GAConfig.h>

using namespace std;
using namespace pqxx;

class Individual : public vector<bool> {
public:
    Individual(int expectedSize, const string& chromosome = "");
    Individual(const Individual& src);
    void print() const;
    void fullPrint() const;
    string to_string() const;

    void crossoverWith(Individual &other);
    void mutate();
    bool paredoDominates(const Individual &opponent) const;
    bool crowdingOperator(const Individual& opponent) const;
    bool operator==(const Individual &opponent) const;
    Individual& operator=(const Individual &other);

    void decodeChromosome();
    void evaluate();

    CapsNetConfig capsNetConfig;
    double loss_100, loss_300;
    double accuracy_100, accuracy_300;

    double crowdingDistance;
    unsigned int rank;

    vector<Individual*> individualsIDominate;
    unsigned int numDominateMe = 0;

private:
    void generateRandom();
};

class CapsuleNetworkDAO {
public:
    CapsuleNetworkDAO();
    void run();

private:
    string buildQuery();
};

#endif //NEURALNETS_INDIVIDUAL_H