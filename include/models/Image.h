//
// Created by Daniel Lopez on 12/29/17.
//

#ifndef NEURALNETS_IMAGE_H
#define NEURALNETS_IMAGE_H

#include <vector>
#include "FeatureMap.h"

using namespace std;

class Image : public vector<unsigned char> {
public:
    Image();
    void addRow(const vector<unsigned char> &row);
    void print() const;
    vector<double> toVectorOfDoubles() const;
    FeatureMap to2DImage() const;

    size_t getLabel() const;
    void setLabel(unsigned char l);
private:
    unsigned char label;
};


#endif //NEURALNETS_IMAGE_H
