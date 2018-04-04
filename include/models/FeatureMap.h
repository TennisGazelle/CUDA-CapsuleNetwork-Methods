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
    // from one inputDim (with dimensions expected
    static vector<FeatureMap> toFeatureMaps(size_t h, size_t w, const vector<double> &cubeData);
    void setSize(size_t h, size_t w);
    void clearOut();
    void print() const;
};


#endif //NEURALNETS_FEATUREMAP_H
