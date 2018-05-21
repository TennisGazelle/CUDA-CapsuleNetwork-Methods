//
// Created by Daniel Lopez on 5/4/18.
//

#include <cassert>
#include "Datasets/DataReader.h"

const Image& DataReader::getTrainingImage(int index) const {
    assert (index <= trainingData.size());
    return trainingData[index];
}

const Image& DataReader::getTestingImage(int index) const {
    assert (index <= testingData.size());
    return testingData[index];
}