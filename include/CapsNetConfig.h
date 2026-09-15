//
// Created by daniellopez on 2/5/18.
//

#ifndef NEURALNETS_CONFIG_H
#define NEURALNETS_CONFIG_H

#include <models/ActivationType.h>

/// Historical CapsNet hyperparameters for the MNIST-era architecture.
///
/// Many fields look configurable but large parts of the codebase still assume
/// MNIST 28×28, 10 classes, and related spatial expressions (for example
/// `28-6` / `6*6`). Treat non-default shapes as unproven until tests cover them.
/// Thesis defaults: 3 routing iterations, `double` precision, margin-loss
/// `m_plus`/`m_minus`/`lambda` as below. See docs/ARCHITECTURE.md.
struct CapsNetConfig {
    bool multithreaded = false;

    const int inputHeight = 28, inputWidth = 28;
    const int numClasses = 10;
    const int numEpochs = 50;
    /// Dynamic routing iterations (historical default: 3).
    const int numIterations = 3;
    const ActivationType at = SIGMOID;

    int cnInnerDim = 8, cnOuterDim = 16;
    int cnNumTensorChannels = 2;
    int batchSize = 250;
    double m_plus = 0.9, m_minus = 0.1, lambda = 0.5;

    double learningRate = 0.1;
    CapsNetConfig() = default;
    CapsNetConfig(const CapsNetConfig&) = default;
    CapsNetConfig& operator=(const CapsNetConfig& other);
};

#endif //NEURALNETS_CONFIG_H
