//
// Created by daniellopez on 2/5/18.
//

#include <cmath>
#include <iostream>
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
    std::cout << learningRate << ",";
//    learningRate = 0.01*pow(learningRate_alpha, ++learningRate_t);
    learningRate = -1.65e-9 * learningRate_t++ + 0.1;
}

void Config::resetLearningRate() {
    learningRate_t = 0;
}

double Config::getLearningRate() const {
    return 0.0001;
    return learningRate;
}
