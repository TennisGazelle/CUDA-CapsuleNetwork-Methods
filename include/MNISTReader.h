//
// Created by Daniel Lopez on 12/29/17.
//

#ifndef NEURALNETS_MNISTREADER_H
#define NEURALNETS_MNISTREADER_H

#include <vector>
#include "models/Image.h"

using namespace std;

class MNISTReader {
public:
    ~MNISTReader();
    static MNISTReader* getInstance();
    const Image& getTrainingImage(int index) const;
    const Image& getTestingImage(int index) const;

    vector<Image> trainingData, testingData;
private:
    static MNISTReader* instance;
    MNISTReader() = default;
    void readMNISTData();
    void readDataWithLabels(const string& datafile, const string& labelfile, vector<Image>& dst);

    static inline void grabFromFile(ifstream &fin, unsigned int &num);
};

#endif //NEURALNETS_MNISTREADER_H
