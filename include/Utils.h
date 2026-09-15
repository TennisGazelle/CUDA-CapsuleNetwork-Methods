//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_UTILS_H
#define NEURALNETS_UTILS_H

#include <armadillo>
#include <cstdint>
#include <string>
#include <vector>

#define EPSILON 1e-4

class Utils {
public:
    // All pseudo-random helpers share one engine so corrected experiments can
    // be reproduced from a single seed.
    static void setRandomSeed(std::uint32_t seed);
    static int getRandBetween(int lowerBound, int upperBound);
    static double getRandBetween(double lowerBound, double upperBound);
    static double getWeightRand(double n);

    static int reverseInt(int i);
    static double square_length(const arma::vec &vn);
    static double length(const arma::vec &vn);
    static arma::vec squish(const arma::vec& input);
    static arma::vec safeNormalise(arma::vec input);
    static double getSquashDerivativeLength(const arma::vec &input);

    static std::vector<double> getAsOneDim(const std::vector<arma::vec>& input);
    static std::vector<arma::vec> asCapsuleVectors(int dim, int numVectors, const std::vector<double>& data);

    // Genetic-algorithm helpers.
    static bool randomWithProbability(double prob);
    static int getBinaryAsInt(const std::vector<bool> &subset);
};

#endif //NEURALNETS_UTILS_H
