//
// Created by daniellopez on 2/5/18.
//

#ifndef NEURALNETS_CONFIG_H
#define NEURALNETS_CONFIG_H

#include <models/ActivationType.h>

struct Config {
    bool multithreaded = false;

    int inputHeight = 28, inputWidth = 28;
    int cnInnerDim = 8, cnOuterDim = 16;
    int cnNumTensorChannels = 40;
    int numClasses = 10;

    int batchSize = 250;
    int numEpochs = 600;
    int numIterations = 3;
    double learningRate = 0.01;

    const ActivationType at = SIGMOID;

    int learningRate_t = 0;
    const double learningRate_alpha = 0.9999999;
    double momentum = 0.9;
};

static Config globalConfig;

#endif //NEURALNETS_CONFIG_H
