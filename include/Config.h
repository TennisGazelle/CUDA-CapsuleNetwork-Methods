//
// Created by daniellopez on 2/5/18.
//

#ifndef NEURALNETS_CONFIG_H
#define NEURALNETS_CONFIG_H

class Config {
public:
    static Config* getInstance();
    void updateLearningRate();
    double getLearningRate() const;

    static const int inputHeight = 28, inputWidth = 28;
    static const int batchSize = 250;
    static const int numEpochs = 1000;
    double learningRate = 10;
private:
    Config();
    static Config* instance;
    const double learningRate_alpha = 0.9999;
    int learningRate_t = 0;
    double momentum = 0.9;
};


#endif //NEURALNETS_CONFIG_H
