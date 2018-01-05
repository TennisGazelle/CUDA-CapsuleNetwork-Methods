//
// Created by Daniel Lopez on 12/29/17.
//

#ifndef NEURALNETS_MNISTREADER_H
#define NEURALNETS_MNISTREADER_H

#include <vector>
#include "Image.h"

using namespace std;

class MNISTReader {
public:
    MNISTReader() = default;
    void readMNISTData();
    void readDataWithLabels(const string& datafile, const string& labelfile, vector<Image>& dst);
    static inline void grabFromFile(ifstream &fin, int &num);

    const Image& getTrainingImage(int index) const;
    const Image& getTestingImage(int index) const;
    vector<Image> trainingData, testingData;
};


#endif //NEURALNETS_MNISTREADER_H
