//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_UTILS_H
#define NEURALNETS_UTILS_H

#include <string>
#include <armadillo>

using namespace std;

class Utils {
public:
    static double getRandBetween(double lowerBound, double upperBound);
    static double getWeightRand(double n);
    static int reverseInt(int i);
    static unsigned char reverseChar(char c);
    static long double square_length(const arma::vec &vn);
    static double length(const arma::vec &vn);
    static arma::vec squish(const arma::vec& input);
    static arma::vec safeNormalise(arma::vec input);
    static double getSquashDerivativeLength(const arma::vec &input);

    // the following are inverses of one another // not tested yet
    static vector<double> getAsOneDim(const vector<arma::vec>& input);
    static vector<arma::vec> asCapsuleVectors(int dim, int numVectors, const vector<double>& data);
};


#endif //NEURALNETS_UTILS_H
