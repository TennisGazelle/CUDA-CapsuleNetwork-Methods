//
// Created by daniellopez on 4/4/18.
//

#include <CapsNetConfig.h>
#include <MNISTReader.h>
#include <models/VectorMap.h>
#include <cassert>
#include <Utils.h>
#include <ProgressBar.h>
#include <HostTimer.h>
#include <CUDAUtils.h>
#include "CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h"

CUCapsuleNetwork::CUCapsuleNetwork(const CapsNetConfig& incomingConfig) :
        config(incomingConfig),
        CUPrimaryCaps(incomingConfig, incomingConfig.inputHeight, incomingConfig.inputWidth, incomingConfig.cnNumTensorChannels * incomingConfig.cnInnerDim, 28-6, 28-6) {

    flattenedTensorSize = 6*6*config.cnNumTensorChannels;

    u.resize(config.cnInnerDim * config.numClasses * flattenedTensorSize);
    
    w.resize(config.cnInnerDim * config.cnOuterDim * config.numClasses * flattenedTensorSize);
    w_delta.resize(config.cnInnerDim * config.cnOuterDim * config.numClasses * flattenedTensorSize);
    w_velocity.resize(config.cnInnerDim * config.cnOuterDim * config.numClasses * flattenedTensorSize);
    
    u_hat.resize(config.cnOuterDim * config.numClasses * flattenedTensorSize);
    v.resize(config.numClasses * config.cnOuterDim);
    b.resize(config.numClasses * flattenedTensorSize);
    c.resize(config.numClasses * flattenedTensorSize);
    truth.resize(config.numClasses);
    losses.resize(config.numClasses);
    lengths.resize(config.numClasses);

    w.fillWithRandom();
}

void CUCapsuleNetwork::forwardPropagation(int imageIndex, bool useTraining) {
    Image image;
    if (useTraining) {
        image = MNISTReader::getInstance()->getTrainingImage(imageIndex);
    } else {
        image = MNISTReader::getInstance()->getTestingImage(imageIndex);
    }

    CUPrimaryCaps.setInput(image.toVectorOfDoubles());
    CUPrimaryCaps.forwardPropagate();
    CUPrimaryCaps.squashAndRemapToU(u);

//    u.print("u", config.cnInnerDim * config.numClasses);
    CUUnifiedBlob::CUDA_matrixVectorMultiplication(w, u, u_hat, config.cnInnerDim, config.cnOuterDim, config.numClasses, flattenedTensorSize);
//    u_hat.print("u_hat", config.cnOuterDim);
//    w.print("w", config.cnInnerDim);

    b.CUDA_clear();
    for (int iter = 0; iter < config.numIterations; iter++) {
        CUUnifiedBlob::CUDA_vectorVectorSoftmax(b, c, config.numClasses, flattenedTensorSize);
//        b.print("b", config.numClasses);
//        c.print("c", config.numClasses);

        CUUnifiedBlob::CUDA_weightReduceVectors(u_hat, c, v, config.numClasses, flattenedTensorSize, config.cnOuterDim);
//        u_hat.print("u_hat", config.numClasses * config.cnInnerDim);
//        v.print("v", config.cnOuterDim);

        CUUnifiedBlob::CUDA_vectorSquash(v, config.numClasses * flattenedTensorSize, config.cnOuterDim);
//        v.print("v - activated", config.cnOuterDim);

        CUUnifiedBlob::CUDA_vectorVectorScalarProduct(u_hat, v, b, config.numClasses, flattenedTensorSize, config.cnOuterDim);
//        b.print("b", config.numClasses);
    }
//    v.print("v - final", config.cnOuterDim);
}

double CUCapsuleNetwork::backPropagation(int imageIndex, bool useTraining) {
    size_t label;
    if (useTraining) {
        label = MNISTReader::getInstance()->getTrainingImage(imageIndex).getLabel();
    } else {
        label = MNISTReader::getInstance()->getTestingImage(imageIndex).getLabel();
    }
    truth.CUDA_clear();
    truth.CUDA_setValueAt(label, 1.0);
//    truth.print("truth in BP", config.numClasses);

//    u.print("u", config.numClasses*config.cnInnerDim);
//    cout << "true value: " << label << endl;
//    v.print("v - final", config.cnOuterDim);
    double loss = getLoss();

    CUUnifiedBlob::CUDA_vectorLossFunction(v, truth, config.numClasses, config.cnOuterDim, config.m_plus, config.m_minus, config.lambda);
//    v.print("delta v", config.cnOuterDim);
    cudaDeviceSynchronize();

    CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(w_delta, v, u_hat, config.numClasses, flattenedTensorSize, config.cnInnerDim, config.cnOuterDim);
//    w_delta.print("w delta, one at a time", config.cnInnerDim*config.cnOuterDim);

    CUDAUtils::checkForError("before weighted trans matrix vec mult");
    CUUnifiedBlob::CUDA_scaledDecompositionOfError(v, c, u_hat, config.numClasses, flattenedTensorSize, config.cnOuterDim);
//    u_hat.print("delta_u_hat", config.cnOuterDim * config.numClasses);

    CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(u, w, u_hat, config.numClasses, flattenedTensorSize, config.cnInnerDim, config.cnOuterDim);
//    u.print("delta u", config.cnInnerDim * config.numClasses);

    CUDAUtils::checkForError("before multi vector reduction");
    CUUnifiedBlob::CUDA_multiVectorReduction(u, config.numClasses, flattenedTensorSize, config.cnInnerDim);
//    u.print("delta u, left reduced", config.cnInnerDim * config.numClasses);

    CUDAUtils::checkForError("before squash derivative");
    CUUnifiedBlob::CUDA_vectorSquashDerivative(u, flattenedTensorSize, config.cnInnerDim, config.numClasses);
    u.print("un-squashed delta_u", config.numClasses*config.cnInnerDim);

    CUDAUtils::checkForError("the end of BP in capsnet, before primary caps conv.");
    CUPrimaryCaps.remapErrorToOutput(u);
    CUPrimaryCaps.backpropagate();
    CUDAUtils::checkForError("after BP in primary caps.");
    
//    cout << "after bp in conv. layer" << endl;
    return loss;
}

double CUCapsuleNetwork::getLoss() {
    CUUnifiedBlob::CUDA_getVectorLoss(v, truth, losses, config.numClasses, config.cnOuterDim, config.m_plus, config.m_minus, config.lambda);
    cudaDeviceSynchronize();
    double loss = 0.0;
    for (int i = 0; i < config.numClasses; i++) {
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

    CUUnifiedBlob::CUDA_getSquaredLength(v, lengths, config.numClasses, config.cnOuterDim);
    cudaDeviceSynchronize();
    for (int v_index = 0; v_index < config.numClasses; v_index++) {
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
    auto &data = MNISTReader::getInstance()->testingData;

    ProgressBar pb(data.size());
    for (int i = 0; i < data.size(); i++) {
        forwardPropagation(i, false);
        backPropagation(i, false);
        CUDAUtils::checkForError("CUCapsuleNetwork::runEpoch()");

        if (i % config.batchSize == config.batchSize - 1) {
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
    for (size_t i = 0; i < config.numEpochs; i++) {
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
    cudaDeviceSynchronize();
    CUUnifiedBlob::elementWiseErrorUpdate(w, w_delta, w_velocity, w.getSize());
    CUPrimaryCaps.updateError();
    cudaDeviceSynchronize();
}

void CUCapsuleNetwork::test_detailedFP() {
    vector<double> loss_history;
    for (int i = 0; i < 1; i++) {
        cout << "****" << i << "****" << endl;
//        cout << "another forward prop" << endl;
        forwardPropagation(0);
//        CUUnifiedBlob::CUDA_getSquaredLength(v, lengths, config.numClasses, config.cnOuterDim);
//        v.print("v - original FP", config.cnOuterDim);
//        lengths.print("lengths");

//        cout << "bp" << endl;
        loss_history.push_back(backPropagation(0));
        cout << i << "\t" << loss_history[i] << endl;
//        losses.print("losses");

//        cout << "update" << endl;
        updateWeights();
    }

    for (int i = 0; i < loss_history.size(); i++) {
    	cout << i << "\t" << loss_history[i] << endl;
    }
}