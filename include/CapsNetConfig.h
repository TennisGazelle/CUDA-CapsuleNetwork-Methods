//
// Created by daniellopez on 2/5/18.
//

#ifndef NEURALNETS_CONFIG_H
#define NEURALNETS_CONFIG_H

#include <models/ActivationType.h>

struct CapsNetConfig {
    bool multithreaded = false;

    const int inputHeight = 28, inputWidth = 28;
    const int numClasses = 10;
    const int numEpochs = 50;
    const int numIterations = 3;
    const ActivationType at = SIGMOID;

    int cnInnerDim = 8, cnOuterDim = 16;
    int cnNumTensorChannels = 2;
    int batchSize = 250;
    double m_plus = 0.9, m_minus = 0.1, lambda = 0.5;

    double learningRate = 0.1;
    CapsNetConfig& operator=(const CapsNetConfig& other);
};

#endif //NEURALNETS_CONFIG_H
