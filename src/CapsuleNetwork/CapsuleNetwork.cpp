//
// Created by daniellopez on 2/23/18.
//

#include <Config.h>
#include <MNISTReader.h>
#include <models/VectorMap.h>
#include <Utils.h>
#include <cassert>
#include <cfloat>
#include <HostTimer.h>
#include <ProgressBar.h>
#include "CapsuleNetwork/CapsuleNetwork.h"

CapsuleNetwork::CapsuleNetwork() :
        primaryCaps(Config::inputHeight, Config::inputWidth, 256, 22, 22),
        digitCaps(10, Capsule(8, 16, 1152, 1)) {
    // TODO extrapolate this for varying inputs and outputs
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
    auto vectorMapOutput = VectorMap::toSquishedArrayOfVecs(8, primaryCapsOutput);

    // for each of the digitCaps, make them accept this as input
    vector<arma::vec> outputs(digitCaps.size());
    for (int i = 0; i < digitCaps.size(); i++) {
        outputs[i] = digitCaps[i].forwardPropagate(vectorMapOutput);
    }
    return outputs;
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
    const vector<FeatureMap> primaryCapsOutput = primaryCaps.getOutput();
    auto vectorMapOutput = VectorMap::toSquishedArrayOfVecs(8, primaryCapsOutput);

    // for each of the digitCaps, make them accept this as input
    int bestGuess = 0;
    long double bestLength = 0.0;
    for (int i = 0; i < digitCaps.size(); i++) {
        auto output = digitCaps[i].forwardPropagate(vectorMapOutput);
        long double length = Utils::square_length(output);
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
    }
    cout << "best guess is: " << bestGuess << endl;
    cout << "actual value : " << label << endl;
    if (bestGuess == label) {
        cout << "WE GOT ONE!!!" << endl;
    }

    cout << endl;
}

double CapsuleNetwork::tally(bool useTraining) {
    cout << "tallying..." << endl;
    int numCorrectlyClassified = 0;

    auto& tallyData = MNISTReader::getInstance()->trainingData;
    if (!useTraining) {
        tallyData = MNISTReader::getInstance()->testingData;
    }

    // go through all the datum
    HostTimer timer;
    timer.start();
    for (int i = 0; i < tallyData.size(); i++) {
        auto output = loadImageAndGetOutput(i, useTraining);

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
    }
    timer.stop();

    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "Accuracy (out of " << tallyData.size() << ")       : " << double(numCorrectlyClassified)/double(tallyData.size()) * 100 << endl;
    cout << "                    Time Taken: " << timer.getElapsedTime() << " ms." << endl;
    return double(numCorrectlyClassified)/double(tallyData.size()) * 100;
}

vector<arma::vec> CapsuleNetwork::getErrorGradient(int targetLabel, const vector<arma::vec> &output) {
    vector<arma::vec> error(output.size(), arma::vec(output[0].size(), arma::fill::zeros));
//    error[targetLabel] = normalise(output[targetLabel]);
    error[targetLabel] = normalise(arma::vec(output[0].size(), arma::fill::ones));
    return error;
}

void CapsuleNetwork::runEpoch() {
    auto& data = MNISTReader::getInstance()->trainingData;

    ProgressBar pb(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        vector<arma::vec> output = loadImageAndGetOutput(i);
        vector<arma::vec> error = getErrorGradient(data[i].getLabel(), output);

        backPropagate(error);

        if (i%Config::batchSize == Config::batchSize-1) {
            updateWeights();
            loadImageAndPrintOutput(i);
        }
        pb.updateProgress(i);
    }
}

void CapsuleNetwork::backPropagate(vector<arma::vec> error) {
    assert (error.size() == digitCaps.size());
    assert (error[0].size() == 16);

    vector<arma::vec> primaryCapsError(1152, arma::vec(8, arma::fill::zeros));
    // given the error, put this in the last layer and get the error, and give it to the Conv. net
    for (int i = 0; i < error.size(); i++) {
        auto subset = digitCaps[i].backPropagate(error[i]);
        for (int j = 0; j < 1152; j++) {
            primaryCapsError[i] += subset[i];
        }
    }

    // translate to feature maps
    vector<FeatureMap> convError = VectorMap::toArrayOfFeatureMaps(6, 6, 256, primaryCapsError);
    // give back to the conv net here.
    primaryCaps.backPropagate(convError);
}

double CapsuleNetwork::getTotalMarginLoss(int targetLabel, const vector<arma::vec> &output) const {
    double sumOfLosses = 0.0;
    for (int i = 0; i < output.size(); i++) {
        double loss = getMarginLoss(i == targetLabel, output[i]);
        cout << "loss for " << i << ": " << loss << endl;
        sumOfLosses += loss;
    }
    return sumOfLosses;
}

double CapsuleNetwork::getMarginLoss(bool isPresent, const arma::vec &v_k) const {
    double t_k = isPresent ? 1 : 0;
    const double m_plus = 0.9;
    const double m_minus = 0.1;
    const double lambda = 0.5;
    const double vLength = Utils::length(v_k);

    double lhs = t_k * pow(max(0.0, m_plus - vLength), 2);
    double rhs = lambda * (1 - t_k) * pow(max(0.0, vLength - m_minus), 2);

    return lhs + rhs;
}

void CapsuleNetwork::updateWeights() {
    primaryCaps.updateError();
    for (auto& cap : digitCaps) {
        cap.updateWeights();
    }
}

void CapsuleNetwork::train() {
    vector<double> history;
    for (size_t i = 0; i < Config::numEpochs; i++) {
        cout << "EPOCH ITERATION: " << i << endl;
        runEpoch();
        history.push_back(tally(false));

        // TODO file writing (and eventual reading)

        cout << endl;
    }

    for (int i = 0; i < history.size(); i++) {
        cout << i << "," << history[i] << endl;
    }
    cout << "DONE!" << endl;
}