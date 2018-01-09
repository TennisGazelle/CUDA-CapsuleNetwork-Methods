//
// Created by Daniel Lopez on 1/8/18.
//

#ifndef NEURALNETS_FILTER_H
#define NEURALNETS_FILTER_H

#include <vector>

using namespace std;

class Filter : public vector< vector<double> > {
public:
    void clearOut();
};


#endif //NEURALNETS_FILTER_H
