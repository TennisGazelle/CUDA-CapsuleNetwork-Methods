//
// Created by Daniel Lopez on 12/29/17.
//

#ifndef NEURALNETS_MNISTREADER_H
#define NEURALNETS_MNISTREADER_H

#include <models/Image.h>

#include <string>
#include <vector>

class MNISTReader {
public:
    ~MNISTReader() = default;
    static MNISTReader* getInstance();

    Image getTrainingImage(int index) const;
    Image getTestingImage(int index) const;
    Image* getTrainingImageRef(int index);
    Image* getTestingImageRef(int index);

    std::vector<Image> trainingData, testingData;

private:
    MNISTReader();
    void readMNISTData();
    static std::string resolveDataDirectory();
};

#endif //NEURALNETS_MNISTREADER_H
