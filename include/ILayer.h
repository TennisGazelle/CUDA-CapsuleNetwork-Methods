//
// Created by Daniel Lopez on 12/28/17.
//

#ifndef NEURALNETS_ILAYER_H
#define NEURALNETS_ILAYER_H

#include <vector>

using namespace std;

class ILayer {
public:
    ILayer() = default;

    virtual void setInput(const vector<double> pInput);
    virtual vector<double> const& getOutput() const;

protected:
    vector<double> input, output;
    size_t inputSize, outputSize;
};


#endif //NEURALNETS_ILAYER_H
