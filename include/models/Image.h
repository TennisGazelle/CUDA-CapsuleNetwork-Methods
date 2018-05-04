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
    explicit Image(int height = 28, int width = 28);
    Image(size_t label, const vector<double>& input);
    void addRow(const vector<unsigned char> &row);
    void print() const;
    vector<double> toVectorOfDoubles() const;
    void fromVectorOfUnsignedChars(const vector<unsigned char> &input);
    FeatureMap toFeatureMap() const;

    size_t getLabel() const;
    void setLabel(unsigned char l);
private:
    unsigned char label;
    int height, width;
};


#endif //NEURALNETS_IMAGE_H
