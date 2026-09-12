//
// Created by daniellopez on 6/7/18.
//

#include <CapsNetConfig.h>

CapsNetConfig& CapsNetConfig::operator=(const CapsNetConfig &other) {
    multithreaded = other.multithreaded;
    cnInnerDim = other.cnInnerDim;
    cnOuterDim = other.cnOuterDim;
    cnNumTensorChannels = other.cnNumTensorChannels;
    batchSize = other.batchSize;
    m_plus = other.m_plus;
    m_minus = other.m_minus;
    lambda = other.lambda;
    learningRate = other.learningRate;
    return *this;
}