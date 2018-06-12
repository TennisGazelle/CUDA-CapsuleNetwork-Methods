//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_UTILS_H
#define NEURALNETS_UTILS_H

#include <string>
#include <armadillo>

class Utils {
public:
    static int getRandBetween(int lowerBound, int upperBound);
    static double getRandBetween(double lowerBound, double upperBound);
    static double getWeightRand(double n);
    static int reverseInt(int i);
    static long double square_length(const arma::vec &vn);
    static double length(const arma::vec &vn);
    static arma::vec squish(const arma::vec& input);
    static arma::vec safeNormalise(arma::vec input);
    static double getSquashDerivativeLength(const arma::vec &input);

    // the following are inverses of one another // not tested yet
    static std::vector<double> getAsOneDim(const std::vector<arma::vec>& input);
    static std::vector<arma::vec> asCapsuleVectors(int dim, int numVectors, const std::vector<double>& data);

    // things for GA
    static bool randomWithProbability(double prob);
    static int getBinaryAsInt(const std::vector<bool> &subset);
};


#endif //NEURALNETS_UTILS_H
