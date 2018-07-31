//
// Created by Daniel Lopez on 12/29/17.
//

#ifndef NEURALNETS_MNISTREADER_H
#define NEURALNETS_MNISTREADER_H

#include <vector>
#include <string>
#include "models/Image.h"

using namespace std;

class MNISTReader {
public:
    ~MNISTReader();
    static MNISTReader* getInstance();
    Image getTrainingImage(int index) const;
    Image getTestingImage(int index) const;
    Image* getTrainingImageRef(int index);
    Image* getTestingImageRef(int index);

    vector<Image> trainingData, testingData;
private:
    static MNISTReader* instance;
    MNISTReader() = default;
    void readMNISTData();
    void readDataWithLabels(const string& datafile, const string& labelfile, vector<Image>& dst);

    static inline void grabFromFile(ifstream &fin, unsigned int &num);
};

#endif //NEURALNETS_MNISTREADER_H
