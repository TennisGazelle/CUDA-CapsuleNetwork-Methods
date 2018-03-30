//
// Created by daniellopez on 2/5/18.
//

#ifndef NEURALNETS_CONFIG_H
#define NEURALNETS_CONFIG_H

#include <models/ActivationType.h>

class Config {
public:
    static Config* getInstance();
    void updateLearningRate();
    double getLearningRate() const;
    void resetLearningRate();

    static const int inputHeight = 28, inputWidth = 28;
    static const int batchSize = 250;
    static const int numEpochs = 1000;
    double learningRate = 0.01;
    const ActivationType at = SIGMOID;
    static const bool multithreaded = true;
private:
    Config();
    static Config* instance;
    const double learningRate_alpha = 0.9999999;
    int learningRate_t = 0;
    double momentum = 0.9;
};


#endif //NEURALNETS_CONFIG_H
