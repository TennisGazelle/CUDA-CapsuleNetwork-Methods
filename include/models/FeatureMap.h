//
// Created by Daniel Lopez on 1/4/18.
//

#ifndef NEURALNETS_FEATUREMAP_H
#define NEURALNETS_FEATUREMAP_H

#include <vector>

using namespace std;

class FeatureMap : public vector< vector<double> > {
public:
    // to one dimension
    vector<double> toOneDim() const;
    // from one dim (with dimensions expected
    FeatureMap& toFeatureMap(size_t h, size_t w) const;
};


#endif //NEURALNETS_FEATUREMAP_H
