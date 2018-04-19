//
// Created by daniellopez on 4/4/18.
//

#include <Config.h>
#include "CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h"

CUCapsuleNetwork::CUCapsuleNetwork() : primaryCaps(Config::inputHeight, Config::inputWidth, Config::cnNumTensorChannels * Config::cnInnerDim, 28-6, 28-6) {
    flattenedTensorSize = 6*6*Config::cnNumTensorChannels;

    u.resize(Config::cnInnerDim * Config::numClasses * flattenedTensorSize);
    w.resize(Config::cnInnerDim * Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    w_delta.resize(Config::cnInnerDim * Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    w_velocity.resize(Config::cnInnerDim * Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    u_hat.resize(Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    v.resize(Config::numClasses * Config::cnOuterDim);
    b.resize(Config::numClasses * flattenedTensorSize);
    c.resize(Config::numClasses * flattenedTensorSize);
}

