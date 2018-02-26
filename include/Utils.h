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
    static long double square_length(const arma::vec &vn);
    static double length(const arma::vec &vn);
    static arma::vec squish(const arma::vec& input);
};


#endif //NEURALNETS_UTILS_H
