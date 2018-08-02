//
// Created by Daniel Lopez on 5/4/18.
//

#ifndef NEURALNETS_DATAREADER_H
#define NEURALNETS_DATAREADER_H

#include <models/Image.h>

class DataReader {
public:
    const Image& getTrainingImage(int index) const;
    const Image& getTestingImage(int index) const;

    vector<Image> trainingData, testingData;

private:
    virtual void readData() = 0;
    virtual void readDataWithLabels(const string &datafile, const string &labelfile, vector<Image> &dst) = 0;
};


#endif //NEURALNETS_DATAREADER_H
