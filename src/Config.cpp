//
// Created by daniellopez on 2/5/18.
//

#include <cmath>
#include "Config.h"

Config* Config::instance = nullptr;

Config::Config() = default;

Config* Config::getInstance() {
    if (instance == nullptr) {
        instance = new Config;
    }
    return instance;
}

void Config::updateLearningRate() {
    learningRate = 0.001 * pow(learningRate_alpha, ++learningRate_t);
}

double Config::getLearningRate() const {
    return learningRate;
}

double Config::getMomentum() const {
    return momentum;
}

int Config::getNumEpochs() const {
    return numEpochs;
}