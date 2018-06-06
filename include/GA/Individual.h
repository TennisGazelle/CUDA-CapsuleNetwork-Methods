//
// Created by daniellopez on 6/4/18.
//

#ifndef NEURALNETS_INDIVIDUAL_H
#define NEURALNETS_INDIVIDUAL_H

#include <vector>
#include <string>
#include <CapsNetConfig.h>
#include <GAConfig.h>

using namespace std;

class Individual : public vector<bool> {
public:
    Individual(int expectedSize, const string& chromosome = "");
    void print() const;
    void fullPrint() const;
    string to_string() const;

    void crossoverWith(Individual &other);
    void mutate();
    bool paretoDominates(const Individual &opponent) const;
    bool operator==(const Individual &opponent) const;

    void decodeChromosome();
    void evaluate();

    CapsNetConfig capsNetConfig;
    double loss_100, loss_300;
    double accuracy_100, accuracy_300;

    vector<Individual*> individualsIDominate;

private:
    void generateRandom();
};


#endif //NEURALNETS_INDIVIDUAL_H
