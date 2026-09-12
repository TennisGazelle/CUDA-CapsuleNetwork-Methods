//
// Created by daniellopez on 2/23/18.
//

#include <CapsNetConfig.h>
#include <MNISTReader.h>
#include <models/VectorMap.h>
#include <Utils.h>
#include <cassert>
#include <HostTimer.h>
#include <ProgressBar.h>
#include <thread>
#include "CapsuleNetwork/CapsuleNetwork.h"

CapsuleNetwork::CapsuleNetwork(const CapsNetConfig& incomingConfig) :
        config(incomingConfig),
        primaryCaps(incomingConfig.inputHeight, incomingConfig.inputWidth,
                    incomingConfig.cnNumTensorChannels * incomingConfig.cnInnerDim,
                    28 - 6,
                    28 - 6),
        digitCaps((unsigned int)incomingConfig.numClasses),
        reconstructionLayers(incomingConfig,
                             incomingConfig.numClasses * incomingConfig.cnOuterDim,
                             incomingConfig.inputHeight * incomingConfig.inputWidth,
                             {28 * 28}) {
    auto totalNumVectors = 6 * 6 * incomingConfig.cnNumTensorChannels;
    assert (digitCaps.size() == incomingConfig.numClasses);
    for (auto &capsule : digitCaps) {
        capsule.init(incomingConfig.cnInnerDim, incomingConfig.cnOuterDim, totalNumVectors, incomingConfig.numClasses, incomingConfig.numIterations);
    }
    reconstructionLayers.init();
}

CapsuleNetwork::~CapsuleNetwork() {
    for (auto& v : interimOutput) {
        v.reset();
    }
    for (auto& v : interimError) {
        v.reset();
    }
}

vector<arma::vec> CapsuleNetwork::loadImageAndGetOutput(int imageIndex, bool useTraining) {
    FeatureMap image;
    if (useTraining) {
        image = MNISTReader::getInstance()->getTrainingImage(imageIndex).toFeatureMap();
    } else {
        image = MNISTReader::getInstance()->getTestingImage(imageIndex).toFeatureMap();
    }

    primaryCaps.setInput({image});
    primaryCaps.calculateOutput();
    vector<FeatureMap> primaryCapsOutput = primaryCaps.getOutput();
    vector<arma::vec> vectorMapOutput = VectorMap::toSquishedArrayOfVecs(config.cnInnerDim, primaryCapsOutput);

    // for each of the digitCaps, make them accept this as input
    vector<arma::vec> outputs(digitCaps.size());

    if (config.multithreaded) {
        static thread workers[10];
        for (int i = 0; i < digitCaps.size(); i++) {
            workers[i] = thread(&CapsuleNetwork::m_threading_loadCapsuleAndGetOutput, this, i, vectorMapOutput);
        }
        // allocate a thread for each one of these
        for (int i = 0; i < digitCaps.size(); i++) {
            workers[i].join();
            outputs[i] = digitCaps[i].getOutput();
        }
    } else {
        for (int i = 0; i < digitCaps.size(); i++) {
            digitCaps[i].forwardPropagate(vectorMapOutput);
            outputs[i] = digitCaps[i].getOutput();
        }
    }
    return outputs;
}

void CapsuleNetwork::m_threading_loadCapsuleAndGetOutput(int capsuleIndex, const vector<arma::vec> &input) {
    digitCaps[capsuleIndex].forwardPropagate(input);
}

vector<arma::vec> CapsuleNetwork::getReconstructionError(vector<arma::vec> digitCapsOutput, int imageIndex, bool useTraining) {
    Image image;
    if (useTraining) {
        image = MNISTReader::getInstance()->getTrainingImage(imageIndex);
    } else {
        image = MNISTReader::getInstance()->getTestingImage(imageIndex);
    }

    // check the label, and zero out all the other capsule outputs
    for (int i = 0; i < digitCapsOutput.size(); i++) {
        if (i != image.getLabel()) {
            digitCapsOutput[i].zeros();
        }
    }

    auto reconstructionImage = reconstructionLayers.loadInputAndGetOutput(Utils::getAsOneDim(digitCapsOutput));
    auto reconstructionGradient = getErrorGradientImage(image, reconstructionImage);
    auto mlpError = Utils::asCapsuleVectors(16, 10, reconstructionLayers.backPropagateError(reconstructionGradient));
    for (auto &v : mlpError) {
        v *= 0.005; // to not dominate as other error
    }
    return mlpError;
}

void CapsuleNetwork::loadImageAndPrintOutput(int imageIndex, bool useTraining) {
    FeatureMap image;
    size_t label;
    if (useTraining) {
        image = MNISTReader::getInstance()->getTrainingImage(imageIndex).toFeatureMap();
        label = MNISTReader::getInstance()->getTrainingImage(imageIndex).getLabel();
    } else {
        image = MNISTReader::getInstance()->getTestingImage(imageIndex).toFeatureMap();
        label = MNISTReader::getInstance()->getTestingImage(imageIndex).getLabel();
    }

    primaryCaps.setInput({image});
    primaryCaps.calculateOutput();
    auto primaryCapsOutput = primaryCaps.getOutput();
    auto vectorMapOutput = VectorMap::toSquishedArrayOfVecs(config.cnInnerDim, primaryCapsOutput);

    // for each of the digitCaps, make them accept this as input
    int bestGuess = 0;
    long double bestLength = 0.0;
    vector<arma::vec> digitCapsOutput(10);
    for (int i = 0; i < digitCaps.size(); i++) {
        digitCapsOutput[i] = digitCaps[i].forwardPropagate(vectorMapOutput);
        long double length = Utils::square_length(digitCapsOutput[i]);
        cout << i << " vector length: " << length;

        if (bestLength < length) {
            bestLength = length;
            bestGuess = i;
            cout << "*";
        }
        if (i == label) {
            cout << "<==";
        }

        cout << endl;
        // get the reconstruction and print it out
    }
    cout << "best guess is: " << bestGuess << endl;
    cout << "actual value : " << label << endl;
    if (bestGuess == label) {
        cout << "WE GOT ONE!!!" << endl;
    }
}

pair<double, long double> CapsuleNetwork::tally(bool useTraining) {
    cout << "tallying..." << endl;
    int numCorrectlyClassified = 0;
    long double totalLoss = 0.0;

    auto &tallyData = MNISTReader::getInstance()->trainingData;
    if (!useTraining) {
        tallyData = MNISTReader::getInstance()->testingData;
    }

    // go through all the datum
    ProgressBar pb(tallyData.size());
    HostTimer timer;
    timer.start();
    for (int i = 0; i < tallyData.size(); i++) {
        auto output = loadImageAndGetOutput(i, useTraining);
        totalLoss += getTotalMarginLoss(tallyData[i].getLabel(), output);

        int guess = 0;
        long double longestVector = 0.0;

        // find the longest vector
        for (int j = 0; j < output.size(); j++) {
            long double length = Utils::square_length(output[j]);
            if (longestVector < length) {
                longestVector = length;
                guess = j;
            }
        }

        // if this matches the label, success for image
        if (guess == tallyData[i].getLabel()) {
            numCorrectlyClassified++;
        }
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

vector<arma::vec> CapsuleNetwork::getErrorGradient(const vector<arma::vec> &output, int targetLabel) {
    vector<arma::vec> error(output.size());
    // generate the derivative of the non-linear vector activation function
    for (int i = 0; i < error.size(); i++) {
        // d(squash())/dv
        auto activationDerivativeLength = Utils::getSquashDerivativeLength(output[i]);
        // Note: this is the first derivative of the activation function
        // d(loss(v))/d||v||
        auto errorGradient = getMarginLossGradient(i == targetLabel, output[i]);
        // loss(v)
        auto rawMarginLoss = getMarginLoss(i == targetLabel, output[i]);
        error[i] = config.learningRate * activationDerivativeLength * errorGradient * rawMarginLoss * normalise(output[i]);

//        printf("raw sum of squares: %f\n", Utils::square_length(output[i]));
//        printf("for class %d, length: %f, t_k: %i, activation: %f, gradient: %f, loss: %f\n",
//               i, Utils::length(output[i]), i == targetLabel, activationDerivativeLength, errorGradient, rawMarginLoss);
    }
    return error;
}

void CapsuleNetwork::runEpoch() {
    auto &data = MNISTReader::getInstance()->trainingData;

    ProgressBar pb(data.size());
    for (int i = 0; i < data.size(); i++) {
        interimOutput = loadImageAndGetOutput(i);
        interimError = getErrorGradient(interimOutput, data[i].getLabel());
        backPropagate();

//        interimError = getReconstructionError(output, i);
//        backPropagate();

        if (i % config.batchSize == config.batchSize - 1) {
            updateWeights();
        }
        pb.updateProgress(i);
    }
}

void CapsuleNetwork::backPropagate() {
    backPropagate(interimError);
}

void CapsuleNetwork::backPropagate(const vector<arma::vec>& error) {
    assert (interimError.size() == digitCaps.size());
    assert (interimError[0].size() == config.cnOuterDim);

    auto flattenedTensorSize = 6 * 6 * config.cnNumTensorChannels;

    vector<arma::vec> primaryCapsError(flattenedTensorSize, arma::vec(config.cnInnerDim, arma::fill::zeros));
    // given the error, put this in the last layer and get the error, and give it to the Conv. net
    for (int i = 0; i < interimError.size(); i++) {
        vector<arma::vec> subset = digitCaps[i].backPropagate(interimError[i]);
        for (int j = 0; j < flattenedTensorSize; j++) {
            primaryCapsError[i] += subset[i];
        }
    }
    for (auto &delta_u : primaryCapsError) {
        auto derivativeLength = Utils::getSquashDerivativeLength(delta_u);
        delta_u = derivativeLength * Utils::safeNormalise(delta_u);
    }
    // translate to feature maps
    vector<FeatureMap> convError = VectorMap::toArrayOfFeatureMaps(6, 6,
                                                                   config.cnNumTensorChannels * config.cnInnerDim,
                                                                   primaryCapsError);
    // give back to the conv net here.
    primaryCaps.backPropagate(convError);
}

long double CapsuleNetwork::getTotalMarginLoss(int targetLabel, const vector<arma::vec> &output) const {
    long double sumOfLosses = 0.0;
    for (int i = 0; i < output.size(); i++) {
        long double loss = getMarginLoss(i == targetLabel, output[i]);
        sumOfLosses += loss;
    }
    return sumOfLosses;
}

double CapsuleNetwork::getMarginLoss(bool isPresent, const arma::vec &v_k) const {
    const double vLength = Utils::length(v_k);

    if (isPresent) {
        return pow(max(0.0, config.m_plus - vLength), 2);
    } else {
        return config.lambda * pow(max(0.0, vLength - config.m_minus), 2);
    }
}

double CapsuleNetwork::getMarginLossGradient(bool isPresent, const arma::vec &v_k) const {
    double t_k = isPresent ? 1.0 : 0.0;
    const double vector_length = Utils::length(v_k);

    double value;
    if (vector_length < config.m_plus) {
        if (vector_length <= config.m_minus) {
            value = -2 * t_k * (config.m_plus - vector_length);
        } else {
            value = 2 * ((config.lambda * (t_k - 1) * (config.m_minus - vector_length)) + t_k * (vector_length - config.m_plus));
        }
    } else {
        value = 2 * config.lambda * (t_k - 1) * (config.m_minus - vector_length);
    }
    return value;
}

void CapsuleNetwork::updateWeights() {
    primaryCaps.updateError();
    for (auto &cap : digitCaps) {
        cap.updateWeights();
    }
//    reconstructionLayers.batchUpdate();
}

void CapsuleNetwork::train() {
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

vector<double> CapsuleNetwork::getErrorGradientImage(const Image &truth, const vector<double> &networkOutput) {
    vector<double> gradient(truth.size());
    for (int i = 0; i < gradient.size(); i++) {
        gradient[i] = networkOutput[i] * (1 - networkOutput[i]) * (truth[i] - networkOutput[i]);
    }
    return gradient;
}

void CapsuleNetwork::fullForwardPropagation(int imageIndex) {
    interimOutput = loadImageAndGetOutput(imageIndex, true);
//    auto reconstructionImage = reconstructionLayers.loadInputAndGetOutput(Utils::getAsOneDim(output));
}

void CapsuleNetwork::fullBackwardPropagation(int imageIndex) {
    auto &data = MNISTReader::getInstance()->trainingData;
    interimError = getErrorGradient(interimOutput, data[imageIndex].getLabel());
    backPropagate();
}

void CapsuleNetwork::verificationTest() {
    int imageIndex = 1;
    FeatureMap image;
    image = MNISTReader::getInstance()->getTrainingImage(imageIndex).toFeatureMap();

    primaryCaps.setInput({image});
    primaryCaps.calculateOutput();
    vector<FeatureMap> primaryCapsOutput = primaryCaps.getOutput();
//    cout << "conv. output" << endl;
//    for (auto& map : primaryCapsOutput) {
//        map.print();
//    }
//    cout << endl;

    vector<arma::vec> vectorMapOutput = VectorMap::toSquishedArrayOfVecs(config.cnInnerDim, primaryCapsOutput);

//    cout << "u vector..." << endl;
//    for (auto& v : vectorMapOutput) {
//        for (int i = 0; i < config.cnInnerDim; i++) {
//            cout << v[i] << "\t";
//        }
//        cout << endl;
//    }
//    cout << endl;

    // for each of the digitCaps, make them accept this as input
    cout << "u_hats" << endl;
    for (int i = 0; i < digitCaps.size(); i++) {
        digitCaps[i].forwardPropagate(vectorMapOutput);
    }

    cout << "capsule outputs" << endl;
    vector<arma::vec> outputs(digitCaps.size());
    for (int i = 0; i < digitCaps.size(); i++) {
        outputs[i] = digitCaps[i].getOutput();
//        outputs[i].t().print();
        for (int j = 0; j < outputs[i].size(); j++) {
            cout << outputs[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    // BACK PROPAGATION
    cout << "\\delta \\mathbf{v}" << endl;
    auto error = getErrorGradient(outputs, MNISTReader::getInstance()->getTrainingImage(imageIndex).getLabel());
    for (int i = 0; i < digitCaps.size(); i++) {
//        error[i].t().print();
        for (int j = 0; j < error[i].size(); j++) {
            cout << error[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    auto flattenedTensorSize = 6 * 6 * config.cnNumTensorChannels;
    vector<arma::vec> delta_u(flattenedTensorSize, arma::vec(config.cnInnerDim, arma::fill::zeros));
    // given the error, put this in the last layer and get the error, and give it to the Conv. net
    for (int i = 0; i < error.size(); i++) {
        vector<arma::vec> subset = digitCaps[i].backPropagate(error[i]);
        for (int j = 0; j < subset.size(); j++) {
            delta_u[i] += subset[i];
        }

//        if (i == 5) {
//            cout << "\\delta \\mathbf{u}" << endl;
//            for (int j = 0; j < flattenedTensorSize; j++) {
//                subset[j].t().print();
//            }
//        }
    }

//    cout << endl;
//    cout << "w - delta" << endl;
//    for (int i = 0; i < digitCaps.size(); i++) {
//        cout << "class: " << i << endl;
//        for (int r = 0; r < config.cnOuterDim; r++) {
//            for (int c = 0; c < config.cnInnerDim; c++) {
//                cout << digitCaps[i].weightDeltas[0].at(r, c) << "\t";
//            }
//            cout << endl;
//        }
//    }

    for (auto &delta_u_vec : delta_u) {
        auto derivativeLength = Utils::getSquashDerivativeLength(delta_u_vec);
        delta_u_vec = derivativeLength * Utils::safeNormalise(delta_u_vec);
    }

    cout << "delta u vectors..." << endl;
    for (auto &delta_u_vec : delta_u) {
//        delta_u_vec.t().print();
        for (int i = 0; i < delta_u_vec.size(); i++) {
            cout << delta_u_vec[i] << "\t";
        }
        cout << endl;
    }


    // translate to feature maps
    vector<FeatureMap> convError = VectorMap::toArrayOfFeatureMaps(6, 6,
                                                                   config.cnNumTensorChannels * config.cnInnerDim,
                                                                   delta_u);
    cout << endl;
    cout << "delta u transformed as feature maps (for conv. output)" << endl;
    for (const auto &fm : convError) {
        fm.print();
    }

    // give back to the conv net here.
    primaryCaps.backPropagate(convError);

    cout << "delta filters..." << endl;
//    for (int f = 0; f < primaryCaps.filterAdjustments.size(); f++) {
//        for (int ch = 0; ch < primaryCaps.filterAdjustments[f].size(); ch++) {
//            for (int r = 0; r < primaryCaps.filterAdjustments[f][ch].size(); r++) {
//                for (int c = 0; c < primaryCaps.filterAdjustments[f][ch][r].size(); c++) {
//                    cout << primaryCaps.filterAdjustments[f][ch][r][c] << "\t";
//                }
//                cout << endl;
//            }
//        }
//    }
}