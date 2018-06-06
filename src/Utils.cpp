//
// Created by Daniel Lopez on 12/28/17.
//

#include <random>
#include <cassert>

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
    static normal_distribution<> dis(0, 1);

    return dis(gen)/100.0;
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

long double Utils::square_length(const arma::vec &vn) {
    long double sum = 0.0;
    for (auto& v : vn) {
        sum += pow(v, 2);
    }
    return sum + 1e-4;
}

double Utils::length(const arma::vec &vn) {
    return (double) sqrt(square_length(vn) + 1e-4);
}

double Utils::getSquashDerivativeLength(const arma::vec &input) {
    auto l = length(input);
    return (2*l) / pow(l*l + 1,2);
}

arma::vec Utils::squish(const arma::vec &input) {
    auto lengthSquared = Utils::square_length(input);
    auto squishingScalar = lengthSquared / (1 + lengthSquared);
//    return squishingScalar * normalise(input, 1);
    return squishingScalar * safeNormalise(input);
}

arma::vec Utils::safeNormalise(arma::vec input) {
    auto l = length(input);
    if (l == 0.0) {
        return input; // a zero vector is itself
    }
    return input / l;
}

vector<double> Utils::getAsOneDim(const vector<arma::vec> &input) {
    vector<double> result;
    result.reserve(input.size() * input[0].size());
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[i].size(); j++) {
            result.push_back(input[i][j]);
        }
    }
    return result;
}

vector<arma::vec> Utils::asCapsuleVectors(int dim, int numVectors, const vector<double> &data) {
    assert (data.size() <= dim*numVectors);
    vector<arma::vec> result(numVectors, arma::vec(dim, arma::fill::zeros));
    for (int v = 0; v < numVectors; v++) {
        for (int d = 0; d < dim; d++) {
            result[v][d] = data[(v*dim)+d];
        }
    }
    return result;
}

bool Utils::randomWithProbability(double prob) {
	double shot = double(rand())/double(RAND_MAX);
	return (shot <= prob);
}

int Utils::getBinaryAsInt(const std::vector<bool> &subset) {
	unsigned int power = subset.size()-1;
	unsigned int sum = 0;
	for (unsigned int i = 0; i < subset.size(); i++) {
		sum += subset[i] * pow(2, power--);
	}
	return sum;
}