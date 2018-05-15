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

CUCapsuleNetwork::CUCapsuleNetwork() : CUPrimaryCaps(Config::inputHeight, Config::inputWidth, Config::cnNumTensorChannels * Config::cnInnerDim, 22, 28-6) {
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
    losses.resize(Config::numClasses);
    lengths.resize(Config::numClasses);

    w.fillWithRandom();
}

void CUCapsuleNetwork::forwardPropagation(int imageIndex, bool useTraining) {
    FeatureMap imageFeatureMap;
    Image image;
    if (useTraining) {
        image = MNISTReader::getInstance()->getTrainingImage(imageIndex);
    } else {
        image = MNISTReader::getInstance()->getTestingImage(imageIndex);
    }
    imageFeatureMap = image.toFeatureMap();

    CUPrimaryCaps.setInput(image.toVectorOfDoubles());
    CUPrimaryCaps.forwardPropagate();
    CUPrimaryCaps.squashAndRemapToU(u);

//    u.print("u", Config::cnInnerDim * Config::numClasses);
    CUUnifiedBlob::CUDA_matrixVectorMultiplication(w, u, u_hat, Config::cnInnerDim, Config::cnOuterDim, Config::numClasses, flattenedTensorSize);
//    u_hat.print("u_hat", Config::cnOuterDim);


    for (int iter = 0; iter < Config::numIterations; iter++) {
        CUUnifiedBlob::CUDA_vectorVectorSoftmax(b, c, Config::numClasses, flattenedTensorSize);
//        b.print("b", Config::numClasses);
//        c.print("c", Config::numClasses);

        CUUnifiedBlob::CUDA_weightReduceVectors(u_hat, c, v, Config::numClasses, flattenedTensorSize, Config::cnOuterDim);
//        u_hat.print("u_hat", Config::numClasses * Config::cnInnerDim);
//        v.print("v", Config::cnOuterDim);

        CUUnifiedBlob::CUDA_vectorSquash(v, Config::numClasses * flattenedTensorSize, Config::cnOuterDim);
//        v.print("v - activated", Config::cnOuterDim);

        CUUnifiedBlob::CUDA_vectorVectorScalarProduct(u_hat, v, b, Config::numClasses, flattenedTensorSize, Config::cnOuterDim);
//        b.print("b", Config::numClasses);
    }
//    v.print("v - final", Config::cnOuterDim);

    b.CUDA_clear();
}

void CUCapsuleNetwork::backPropagation(int imageIndex, bool useTraining) {
    size_t label;
    if (useTraining) {
        label = MNISTReader::getInstance()->getTrainingImage(imageIndex).getLabel();
    } else {
        label = MNISTReader::getInstance()->getTestingImage(imageIndex).getLabel();
    }
    truth.CUDA_clear();
    truth.setValueAt_1D(label, 1);

//    u.print("u", Config::numClasses*Config::cnInnerDim);
//    cout << "true value: " << label << endl;
//    v.print("v - final", Config::cnOuterDim);

    CUUnifiedBlob::CUDA_vectorLossFunction(v, truth, Config::numClasses, Config::cnOuterDim);
//    v.print("delta v", Config::cnOuterDim);

    CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(w_delta, v, u, Config::numClasses, flattenedTensorSize, Config::cnInnerDim, Config::cnOuterDim);
//    w_delta.print("delta w, one at a time", Config::cnInnerDim);


    CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(u, c, w, v, Config::numClasses, flattenedTensorSize, Config::cnInnerDim, Config::cnOuterDim);
//    u.print("delta u", Config::cnOuterDim * Config::numClasses);

    CUUnifiedBlob::CUDA_multiVectorReduction(u, Config::numClasses, flattenedTensorSize, Config::cnInnerDim);
//    u.print("delta u, left reduced", Config::cnOuterDim * Config::numClasses);

//    u.print("delta_u", Config::numClasses*Config::cnInnerDim);
//    assert(!std::isnan(u.getValueAt_1D(0)));

    CUUnifiedBlob::CUDA_vectorSquashDerivative(u, flattenedTensorSize, Config::cnInnerDim, Config::numClasses);
//    u.print("un-squashed delta_u", Config::numClasses*Config::cnInnerDim);
    CUPrimaryCaps.remapErrorToOutput(u);
    CUPrimaryCaps.backpropagate();
}

double CUCapsuleNetwork::getLoss() {
    CUUnifiedBlob::getVectorLoss(v, truth, losses, Config::numClasses, Config::cnOuterDim);
    cudaDeviceSynchronize();
    double loss = 0.0;
    for (int i = 0; i < Config::numClasses; i++) {
        loss += losses.getValueAt_1D(i);
    }
    return loss;
}

bool CUCapsuleNetwork::testResults(int imageIndex, bool useTraining) {
    int biggestVectorIndex = 0;
    long double biggestVectorValue = 0.0;

    int trueValue;
    if (useTraining) {
        trueValue = MNISTReader::getInstance()->trainingData[imageIndex].getLabel();
    } else {
        trueValue = MNISTReader::getInstance()->testingData[imageIndex].getLabel();
    }

    CUUnifiedBlob::CUDA_getSquaredLength(v, lengths, Config::numClasses, Config::cnOuterDim);
    cudaDeviceSynchronize();
    for (int v_index = 0; v_index < Config::numClasses; v_index++) {
        double sq_length = lengths.getValueAt_1D(v_index);
        if (sq_length > biggestVectorValue) {
            biggestVectorValue = sq_length;
            biggestVectorIndex = v_index;
        }
    }

    // we have the biggest, make sure it's right
//    cout << "guessed: " << biggestVectorIndex << " actually: " << trueValue << endl;
    return (biggestVectorIndex == trueValue);
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
        forwardPropagation(i, true);
        backPropagation(i, true);

        if (i % Config::batchSize == Config::batchSize - 1) {
            updateWeights();
        }
        pb.updateProgress(i);
    }
}

pair<double, long double> CUCapsuleNetwork::tally(bool useTraining) {
    cout << "tallying..." << endl;
    auto& tallyData = MNISTReader::getInstance()->trainingData;
    if (!useTraining) {
        tallyData = MNISTReader::getInstance()->testingData;
    }

    int numCorrectlyClassified = 0;
    long double totalLoss = 0.0;
    // go through all data points
    HostTimer timer;
    ProgressBar pb(tallyData.size());
    timer.start();
    for (int i = 0; i < tallyData.size(); i++) {
        forwardPropagation(i, useTraining);
        totalLoss += getLoss();
        if (testResults(i, useTraining)) {
            numCorrectlyClassified++;
        }
        backPropagation(i, useTraining);
        pb.updateProgress(i);
    }
    timer.stop();


    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "       Accuracy (out of " << tallyData.size() << "): "
         << double(numCorrectlyClassified) / double(tallyData.size()) * 100 << endl;
    cout << "                    Time Taken: " << timer.getElapsedTime() << " ms." << endl;
    cout << "                  Average Loss: " << totalLoss << endl;
    return {
            double(numCorrectlyClassified) / double(tallyData.size()) * 100,
            totalLoss
    };
}

void CUCapsuleNetwork::train() {
    vector<pair<double, long double>> history;
    for (size_t i = 0; i < Config::numEpochs; i++) {
        cout << "EPOCH ITERATION: " << i << endl;
        runEpoch();
        history.push_back(tally(false));

        // TODO file writing (and eventual reading)

        cout << endl;
        for (int j = 0; j < history.size(); j++) {
            cout << j << ", " << history[j].first << ", " << history[j].second << endl;
        }
    }

    cout << "DONE!" << endl;
}

void CUCapsuleNetwork::updateWeights() {
    CUUnifiedBlob::CUDA_elementWiseErrorUpdate(w, w_delta, w_velocity,
                                               Config::numClasses * flattenedTensorSize * Config::cnInnerDim * Config::cnOuterDim);
    CUPrimaryCaps.updateError();
}