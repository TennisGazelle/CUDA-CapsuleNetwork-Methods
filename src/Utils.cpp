//
// Created by Daniel Lopez on 12/28/17.
//

#include <random>

#include "Utils.h"

using namespace std;

double Utils::getRandBetween(double lowerBound, double upperBound) {
    srand(2);
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<> dis(lowerBound, upperBound);
    return dis(gen);
}

double Utils::getWeightRand(double n) {
    //get a normal distribution centered around 0 [-2.4/n, 2.4/n]
    static random_device rd;
    static mt19937 gen(rd());

    const double distributionHalfWidth = 2.4/n;
    const double stdDev =  distributionHalfWidth * 2 / 6;
    static normal_distribution<> dis(0, stdDev);

    return dis(gen);
}

int Utils::reverseInt(int i) {
    unsigned char
        c1 = i & 255,
        c2 = (i >> 8) & 255,
        c3 = (i >> 16) & 255,
        c4 = (i >> 24) & 255;

    return ((int) c1 << 24) +
           ((int) c2 << 16) +
           ((int) c3 << 8) +
           ((int) c4);
}