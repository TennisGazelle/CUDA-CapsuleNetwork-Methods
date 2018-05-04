//
// Created by Daniel Lopez on 12/29/17.
//

#include <iostream>
#include <cassert>
#include "models/Image.h"

Image::Image(int pHeight, int pWidth) : height(pHeight), width(pWidth) {
    resize(height * width);
}

Image::Image(size_t label, const vector<double> &input) {
    this->label = (unsigned char) label;
    resize(height*width);
    assert (size() == input.size());
    for (int i = 0; i < size(); i++) {
        (*this)[i] = (unsigned char)(input[i] * 256.0);
    }
}

void Image::addRow(const vector<unsigned char> &row) {
    insert(end(), row.begin(), row.end());
}

void Image::print() const {
    int index = 0;
    cout << "A [" << int(label) << "]" << endl;
    for (unsigned int i = 0; i < size(); i++) {
        if (!(i % width)) {
            cout << index++ << ": ";
        }
        cout << (int) at(i) << '\t';
        if (!((i+1) % width)) {
            cout << endl;
        }
    }
}

vector<double> Image::toVectorOfDoubles() const {
    vector<double> result(size());
    for (unsigned int i = 0; i < size(); i++) {
        result[i] = double(at(i));
    }
    assert (result.size() == size());
    return result;
}

void Image::fromVectorOfUnsignedChars(const vector<unsigned char> &input) {
    for (int i = 0; i < size(); i++) {
        (*this)[i] = 255 - input[i];
    }
}

size_t Image::getLabel() const {
    return size_t (label);
}

void Image::setLabel(unsigned char l) {
    label = l;
}

FeatureMap Image::toFeatureMap() const {
    FeatureMap pixels;
    pixels.reserve(28);
    for (unsigned int r = 0; r < 28; r++) {
        vector<double> row(28);
        for (unsigned int col = 0; col < 28; col++) {
            row[col] = double(at(col + (r * 28)))/256.0;
        }
        pixels.push_back(row);
    }

    return pixels;
}