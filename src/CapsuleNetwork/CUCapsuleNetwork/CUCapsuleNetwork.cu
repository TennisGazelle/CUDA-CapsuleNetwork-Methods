//
// Created by daniellopez on 4/4/18.
//

#include <Config.h>
#include <MNISTReader.h>
#include <models/VectorMap.h>
#include <cassert>
#include <Utils.h>
#include <ProgressBar.h>
#include <HostTimer.h>
#include "CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h"

CUCapsuleNetwork::CUCapsuleNetwork() : primaryCaps(Config::inputHeight, Config::inputWidth, Config::cnNumTensorChannels * Config::cnInnerDim, 28-6, 28-6),
                                       other_primaryCaps(Config::inputHeight, Config::inputWidth, Config::cnNumTensorChannels * Config::cnInnerDim, 28-6, 28-6) {
    flattenedTensorSize = 6*6*Config::cnNumTensorChannels;

    u.resize(Config::cnInnerDim * Config::numClasses * flattenedTensorSize);
    w.resize(Config::cnInnerDim * Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    w_delta.resize(Config::cnInnerDim * Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    w_velocity.resize(Config::cnInnerDim * Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    u_hat.resize(Config::cnOuterDim * Config::numClasses * flattenedTensorSize);
    v.resize(Config::numClasses * Config::cnOuterDim);
    b.resize(Config::numClasses * flattenedTensorSize);
    c.resize(Config::numClasses * flattenedTensorSize);
    truth.resize(Config::numClasses);

    w.fillWithRandom();
}

void CUCapsuleNetwork::forwardPropagation(int imageIndex, bool useTraining) {
    FeatureMap image;
    if (useTraining) {
        image = MNISTReader::getInstance()->getTrainingImage(imageIndex).toFeatureMap();
    } else {
        image = MNISTReader::getInstance()->getTestingImage(imageIndex).toFeatureMap();
    }

    primaryCaps.setInput({image});
    primaryCaps.calculateOutput();
    vector<FeatureMap> primaryCapsOutput = primaryCaps.getOutput();
    to1DSquishedArrayOfVecs(Config::cnInnerDim, primaryCapsOutput, u, Config::numClasses);

    CUUnifiedBlob::CUDA_matrixVectorMultiplication(w, u, u_hat, Config::cnInnerDim, Config::cnOuterDim, Config::cnNumTensorChannels*Config::numClasses);

    for (int iter = 0; iter < Config::numIterations; iter++) {
        CUUnifiedBlob::CUDA_vectorVectorSoftmax(b, c, Config::numClasses, flattenedTensorSize);
        CUUnifiedBlob::CUDA_weightReduceVectors(u_hat, c, v, Config::numClasses, flattenedTensorSize, Config::cnOuterDim);
        CUUnifiedBlob::CUDA_vectorSquash(v, Config::numClasses * flattenedTensorSize, Config::cnOuterDim);
        CUUnifiedBlob::CUDA_vectorVectorScalarProduct(u_hat, v, b, Config::numClasses, flattenedTensorSize, Config::cnOuterDim);
    }
}

void CUCapsuleNetwork::backPropagation(int imageIndex, bool useTraining) {
    size_t label;
    if (useTraining) {
        label = MNISTReader::getInstance()->getTrainingImage(imageIndex).getLabel();
    } else {
        label = MNISTReader::getInstance()->getTestingImage(imageIndex).getLabel();
    }
    for (int i = 0; i < Config::numClasses; i++) {
        truth.setValueAt_1D(i, 0);
    }
    truth.setValueAt_1D(label, 1);

    CUUnifiedBlob::CUDA_vectorLossFunction(v, truth, Config::numClasses, Config::cnOuterDim);
    CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(u, c, w, v, Config::numClasses, flattenedTensorSize, Config::cnInnerDim, Config::cnOuterDim);
    CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(w_delta, v, u, Config::numClasses, flattenedTensorSize, Config::cnInnerDim, Config::cnOuterDim);
    CUUnifiedBlob::CUDA_multiVectorReduction(u, Config::numClasses, flattenedTensorSize, Config::cnInnerDim);
    // TODO move error back to format for conv. layer
    CUUnifiedBlob::CUDA_vectorSquashDerivative(u, flattenedTensorSize, Config::cnInnerDim);

}

long double CUCapsuleNetwork::forwardAndBackPropagation(int imageIndex, bool useTraining) {
    forwardPropagation(imageIndex, useTraining);
    backPropagation(imageIndex, useTraining);
    HostTimer ht;
    ht.start();
    updateWeights();
    ht.stop();
    return ht.getElapsedTime();
}

void CUCapsuleNetwork::runEpoch() {
    auto &data = MNISTReader::getInstance()->trainingData;

    ProgressBar pb(data.size());
    for (int i = 0; i < data.size(); i++) {
        CUCapsuleNetwork::forwardAndBackPropagation(i);

        if (i % Config::batchSize == Config::batchSize - 1) {
            updateWeights();
        }
        pb.updateProgress(i);
    }
}

void CUCapsuleNetwork::updateWeights() {
    // TODO finish implementing the momentum updating... (with the w_velocity)
    CUUnifiedBlob::CUDA_matrixMatrixUpdate(w, w_delta, Config::numClasses * flattenedTensorSize * Config::cnInnerDim * Config::cnOuterDim);
}

void CUCapsuleNetwork::to1DSquishedArrayOfVecs(size_t vectorDim, vector<FeatureMap> inputMaps, CUUnifiedBlob &output, int numClasses) const {
    // assuming that the depth of these maps is divisible of the resulting depth (it should be)
    size_t height = inputMaps[0].size();
    size_t width = inputMaps[0][0].size();

    for (size_t r = 0; r < height; r++) {
        for (size_t c = 0; c < width; c++) {
            // make a vector for every "depth" vectors
            arma::vec v(vectorDim);
            // go down the depth of the input at this position
            for (int d = 0; d < inputMaps.size(); d++) {
                if (d%vectorDim == vectorDim-1) {
                    size_t depth_index = (d-(vectorDim-1))/vectorDim;
                    size_t vector_index = r*width*Config::cnNumTensorChannels + c*Config::cnNumTensorChannels + depth_index;

                    // squash and save as I go...
                    v = Utils::squish(v);
                    for (int k = 0; k < numClasses; k++) {
                        for (int i = 0; i < v.size(); i++) {
                            output.setValueAt_2D(vector_index*numClasses*vectorDim + k*vectorDim, i, vectorDim, v[i]);
                        }
                    }
                }
                v[d%vectorDim] = inputMaps[d][r][c];
            }
        }
    }
}