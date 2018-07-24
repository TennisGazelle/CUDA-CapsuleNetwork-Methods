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
        CUPrimaryCaps(incomingConfig,
                      incomingConfig.inputHeight,
                      incomingConfig.inputWidth,
                      incomingConfig.cnNumTensorChannels * incomingConfig.cnInnerDim,
                      28-6, 28-6) {

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

    totalMemoryUsage += config.cnInnerDim * config.numClasses * flattenedTensorSize;
    totalMemoryUsage += 3*(config.cnInnerDim * config.cnOuterDim * config.numClasses * flattenedTensorSize);
    totalMemoryUsage += config.cnOuterDim * config.numClasses * flattenedTensorSize;
    totalMemoryUsage += config.numClasses * config.cnOuterDim;
    totalMemoryUsage += 2*(config.numClasses * flattenedTensorSize);
    totalMemoryUsage += 3*(config.numClasses);
    totalMemoryUsage += CUPrimaryCaps.getTotalMemoryUsage();
    totalMemoryUsage *= 8;
}

void CUCapsuleNetwork::initWithSeq(CapsuleNetwork &originalWeights) {
    //TODO: uncomment this, and include proper getters/setters for each W
    srand(0);
    // populate the w matrices
//    double value;
//    int index = 0;
//    for (int l = 0; l < flattenedTensorSize; l++) {
//    	for (int k = 0; k < config.numClasses; k++) {
//
//            for (int r = 0; r < config.cnOuterDim; r++) {
//                for (int c = 0; c < config.cnInnerDim; c++) {
//                    value = originalWeights.digitCaps[k].weightMatrices[l].at(r, c);
//                    //cout << "original location, capsule: " << k << ", tensor level: " << l << ", coordinates(r,c): (" << r << ", " << c << ") ";
//                    //cout << "index: " << index << " value: " << value << endl;
//                    w.setValueAt_1D(index++, value);
//                }
//            }
//        }
//    }
//
//    // populate the filters from the original
//    index = 0;
//    for (int f = 0; f < originalWeights.primaryCaps.filters.size(); f++) {
//        for (int ch = 0; ch < originalWeights.primaryCaps.filterDepth; ch++) {
//            for (int r = 0; r < originalWeights.primaryCaps.filterHeight; r++) {
//                for (int c = 0; c < originalWeights.primaryCaps.filterWidth; c++) {
//                    value = originalWeights.primaryCaps.filters[f][ch][r][c];
//                    CUPrimaryCaps.filter.setValueAt_1D(index++, value);
//                }
//            }
//        }
//    }
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
//    u_hat.print("u_hat", config.cnOuterDim * config.numClasses);
    if (u.CUDA_hasNan()) {
        cerr << "YELLING, u HAS A NAN; location: " << u.hasNan() << endl;
        u.print("u", config.numClasses * config.cnInnerDim);
        exit(1);
    }
//    if (u.hasInf() != -1) {
//        cerr << "u has an inf" << u.hasInf() << endl;
//        exit(1);
//    }
//    w.print("w", config.cnInnerDim*config.cnOuterDim*config.numClasses);

    b.CUDA_clear();
    for (int iter = 0; iter < config.numIterations; iter++) {
        CUUnifiedBlob::CUDA_vectorVectorSoftmax(b, c, config.numClasses, flattenedTensorSize);
//        b.print("b", config.numClasses);
//        c.print("c", config.numClasses);

        CUUnifiedBlob::CUDA_weightReduceVectors(u_hat, c, v, config.numClasses, flattenedTensorSize, config.cnOuterDim);
//        if (u_hat.isAllZeros()) {
//            u_hat.print("u_hat", config.numClasses * config.cnInnerDim);
//            cerr << "u_hat is all zeros" << endl;
//            exit(1);
//        }
//        if (u_hat.hasInf() != -1) {
//            cerr << "u_hat has an inf: " << u_hat.hasInf() << endl;
//            u.print("u", config.cnInnerDim);
//            w.print("w", config.cnInnerDim * config.cnOuterDim);
//            u_hat.print("u_hat", config.cnOuterDim);
//            exit(1);
//        }
//        v.print("v - iter: " + to_string(iter), config.cnOuterDim);
//        if (v.hasNan() != -1) {
//            cerr << "v HAS A NAN after weight reduction with u_hat and c; location: " << v.hasNan() << ", iteration:" << iter << endl;
//            cerr << "dumping..." << endl;
//            v.print("v", config.cnOuterDim);
//            cerr << "checking u_hat...: " << u_hat.hasNan() << endl;
//            exit(1);
//        }
//        if (v.hasInf() != -1) {
//            cerr << "unsquashed v during routing algorithm: " << v.hasInf() << endl;
//            v.print("squashed with inf", config.cnOuterDim);
//            exit(1);
//        }
        CUUnifiedBlob::CUDA_vectorSquash(v, config.numClasses, config.cnOuterDim);
//        v.print("v - activated", config.cnOuterDim);
        if (v.CUDA_hasNan()) {
            cerr << "v_squashed HAS A NAN; location: " << v.hasNan() << ", iteration:" << iter << endl;
            v.print("squashed v with nan", config.cnOuterDim);
            cache.resize(v.getSize());
            CUUnifiedBlob::CUDA_weightReduceVectors(u_hat, c, cache, config.numClasses, flattenedTensorSize, config.cnOuterDim);
            cache.print("v was originally...", config.cnOuterDim);
            u_hat.print("u_hat is...", config.cnOuterDim * config.numClasses);

            c.print("c", config.numClasses);
            exit(1);
        }

        CUUnifiedBlob::CUDA_vectorVectorScalarProduct(u_hat, v, b, config.numClasses, flattenedTensorSize, config.cnOuterDim);
        if (v.CUDA_hasNan()) {
            cerr << "v_squashed HAS A NAN after vector scalar product with u_hat; location: " << v.hasNan() << ", iteration:" << iter << endl;
            cerr << "checking u_hat...: " << u_hat.hasNan() << endl;
            exit(1);
        }
    }
//    b.print("b", config.numClasses);
//    v.print("v - final", config.cnOuterDim);
//    if (v.isAllZeros()) {
//        cerr << "v is all zeros at the end of a forward prop: here's a dump" << endl;
//        CUPrimaryCaps.printFilter();
//        CUPrimaryCaps.printOutput();
//        CUPrimaryCaps.printInput();
//        u.print("u", config.cnInnerDim * config.numClasses);
//        w.print("w", config.cnInnerDim*config.cnOuterDim*config.numClasses);
//        u_hat.print("u_hat", config.numClasses * config.cnInnerDim);
//        b.print("b", config.numClasses);
//        c.print("c", config.numClasses);
//        exit(1);
//    }
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
    if (v.CUDA_hasNan()) {
        v.print("delta v", config.cnOuterDim);
        cerr << "delta_v HAS A NAN; location: " << v.hasNan() << endl;

        cache.resize(v.getSize());
        cache.copy(v);
        CUUnifiedBlob::CUDA_weightReduceVectors(u_hat, c, cache, config.numClasses, flattenedTensorSize, config.cnOuterDim);
        CUUnifiedBlob::CUDA_vectorSquash(v, config.numClasses, config.cnOuterDim);
        cache.print("v", config.cnOuterDim);

        u_hat.print("original u_hat", config.numClasses * config.cnOuterDim);
        c.print("c", config.numClasses);
        b.print("b", config.numClasses);

        CUPrimaryCaps.printFilter();
        exit(1);
    }
//    v.print("delta v", config.cnOuterDim);

    CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(w_delta, v, u, config.numClasses, flattenedTensorSize, config.cnInnerDim, config.cnOuterDim);
//    w.print("w, one at a time", config.cnInnerDim*config.cnOuterDim*config.numClasses);
//    w_delta.print("w delta, one at a time", config.cnInnerDim*config.cnOuterDim*config.numClasses);

    CUUnifiedBlob::CUDA_scaledDecompositionOfError(v, c, u_hat, config.numClasses, flattenedTensorSize, config.cnOuterDim);
    CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(u, w, u_hat, config.numClasses, flattenedTensorSize, config.cnInnerDim, config.cnOuterDim);
//    b.print("b", config.numClasses);
//    c.print("c", config.numClasses);
//    u_hat.print("delta_u_hat", config.cnOuterDim * config.numClasses);

//    u.print("delta u", config.cnInnerDim * config.numClasses);

    CUUnifiedBlob::CUDA_multiVectorReduction(u, config.numClasses, flattenedTensorSize, config.cnInnerDim);
//    u.print("delta u, left reduced", config.cnInnerDim * config.numClasses);
    if (u.CUDA_hasNan()) {
        cerr << "delta u has a nan: " << u.hasNan() << endl;
        cerr << "dumping..." << endl;
        u.print("left reduced delta_u", config.numClasses*config.cnInnerDim);
        exit(1);
    }

    CUUnifiedBlob::CUDA_vectorSquashDerivative(u, flattenedTensorSize, config.cnInnerDim, config.numClasses);
//    u.print("un-squashed delta_u", config.numClasses*config.cnInnerDim);
//    if (u.CUDA_hasNan()) {
//        cerr << "delta u has a nan after derivative: " << u.hasNan() << endl;
//        cerr << "dumping..." << endl;
//        u.print("unsquashed delta_u", config.numClasses*config.cnInnerDim);
//        exit(1);
//    }

    CUPrimaryCaps.remapErrorToOutput(u);
    CUPrimaryCaps.backPropagate();
    
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

    CUUnifiedBlob::CUDA_getLength(v, lengths, config.numClasses, config.cnOuterDim);
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

long double CUCapsuleNetwork::runEpoch() {
    auto &data = MNISTReader::getInstance()->trainingData;

    long double totalLoss = 0.0;
    ProgressBar pb(data.size());
    for (int i = 0; i < data.size(); i++) {
        forwardPropagation(i);
        totalLoss += backPropagation(i);

        if (i % config.batchSize == config.batchSize - 1) {
            updateWeights();
        }
        pb.updateProgress(i);
    }

    return totalLoss;
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

        if (i % config.batchSize == config.batchSize - 1) {
            updateWeights();
        }
        pb.updateProgress(i);
    }
    timer.stop();


    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "       Accuracy (out of " << tallyData.size() << "): "
         << double(numCorrectlyClassified) / double(tallyData.size()) * 100 << endl;
    cout << "                    Time Taken: " << timer.getElapsedTime() << " ms." << endl;
    cout << "                    Total Loss: " << totalLoss << endl;
    return {
            double(numCorrectlyClassified) / double(tallyData.size()) * 100,
            totalLoss
    };
}

pair<double, long double> CUCapsuleNetwork::train() {
    vector<pair<double, long double>> history;
    for (size_t i = 0; i < config.numEpochs; i++) {
        cout << "EPOCH ITERATION: " << i << endl;
//        runEpoch();
        history.push_back(tally(false));
//        history.push_back({0, runEpoch()});

        // TODO file writing (and eventual reading)

    }
    cout << endl;
    for (int j = 0; j < history.size(); j++) {
        cout << j << "\t" << history[j].first << "\t" << history[j].second << endl;
    }

    cout << "DONE!" << endl;
    return history[history.size()-1];
}

void CUCapsuleNetwork::updateWeights() {
    cudaDeviceSynchronize();
    CUUnifiedBlob::CUDA_elementWiseErrorUpdate(w, w_delta, w_velocity, w.getSize());
    CUPrimaryCaps.updateError();
    cudaDeviceSynchronize();
}

void CUCapsuleNetwork::test_detailedFP() {
    vector< pair<double, bool> > loss_history;
    unsigned int numCorrect = 0, batchSize = 10;

    bool isTrue;
    for (int i = 0; i < 1000; i++) {
        isTrue = false;
        int imageIndex = i%20;

        forwardPropagation(imageIndex);
        CUUnifiedBlob::CUDA_getLength(v, lengths, config.numClasses, config.cnOuterDim);
//        v.print("v - original FP", config.cnOuterDim);

        if (testResults(imageIndex)) {
            isTrue = true;
            numCorrect++;
        }

        loss_history.push_back({backPropagation(imageIndex), isTrue});
        lengths.print("lengths", config.numClasses);
//        v.print("delta_v", config.cnOuterDim);

        if (i%batchSize == batchSize-1) {
//            w.print("w", config.cnInnerDim*config.cnOuterDim);
//            w_delta.print("delta_w", config.cnInnerDim*config.cnOuterDim);
//            u.print("delta u", config.cnInnerDim);
//            losses.print("losses", config.numClasses);
            updateWeights();
        }
    }
//    CUPrimaryCaps.printFilter();

    for (int i = 0; i < loss_history.size(); i++) {
    	cout << i << "\t" << (loss_history[i].second ? "*" : "") << "\t" << loss_history[i].first << endl;
    }
    cout << "num correct: " << numCorrect << endl;
}

void CUCapsuleNetwork::verificationTest() {
    forwardPropagation(1);
//    CUPrimaryCaps.output.print("convolutional layer output", CUPrimaryCaps.outputWidth);
//    u.print("u", config.numClasses * config.cnInnerDim);
//    u_hat.print("u_hat", config.cnOuterDim);
//    v.print("output", config.cnOuterDim);

    // back propagation
    backPropagation(1);
    v.print("delta v", config.cnOuterDim);
//    w_delta.print("w_delta", config.cnInnerDim);
//    u_hat.print("delta u_hat", config.numClasses * config.cnOuterDim);
    u.print("delta u", config.numClasses * config.cnInnerDim);

    CUPrimaryCaps.printOutput();
//    CUPrimaryCaps.filter_error.print("delta filters", CUPrimaryCaps.filterWidth);
}