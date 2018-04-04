//
// Created by Daniel Lopez on 12/28/17.
//

#include "ILayer.h"

void ILayer::setInput(const vector<double> pInput) {
    input = pInput;
}

vector<double> const& ILayer::getOutput() const {
    return output;
}