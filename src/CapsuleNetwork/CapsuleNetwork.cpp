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
#include <thread>
#include "CapsuleNetwork/CapsuleNetwork.h"

CapsuleNetwork::CapsuleNetwork() :
        primaryCaps(Config::inputHeight, Config::inputWidth, 16, 22, 22),
        digitCaps(10),
        reconstructionLayers(10*16, 28*28, {28*28}) {
    // TODO extrapolate this for varying inputs and outputs
    for (auto& capsule : digitCaps) {
        capsule.init(8, 16, 72, 10);
    }
    reconstructionLayers.init();
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
    vector<arma::vec> vectorMapOutput = VectorMap::toSquishedArrayOfVecs(8, primaryCapsOutput);

    // for each of the digitCaps, make them accept this as input
    vector<arma::vec> outputs(digitCaps.size());
    static thread workers[10];
    for (int i = 0; i < digitCaps.size(); i++) {
        workers[i] = thread(&CapsuleNetwork::m_threading_loadCapsuleAndGetOutput, this, i, vectorMapOutput);
    }

    // allocate a thread for each one of these
    for (int i = 0; i < digitCaps.size(); i++) {
        workers[i].join();
        outputs[i] = digitCaps[i].getOutput();
    }
    return outputs;
}

void CapsuleNetwork::m_threading_loadCapsuleAndGetOutput(int capsuleIndex, const vector<arma::vec> input) {
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
    auto vectorMapOutput = VectorMap::toSquishedArrayOfVecs(8, primaryCapsOutput);

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

pair<double, double> CapsuleNetwork::tally(bool useTraining) {
    cout << "tallying..." << endl;
    int numCorrectlyClassified = 0;
    long double totalLoss = 0;

    auto& tallyData = MNISTReader::getInstance()->trainingData;
    if (!useTraining) {
        tallyData = MNISTReader::getInstance()->testingData;
    }

    // go through all the datum
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
    }
    timer.stop();

    cout << "Correctly Classified Instances: " << numCorrectlyClassified << endl;
    cout << "Accuracy (out of " << tallyData.size() << ")       : " << double(numCorrectlyClassified)/double(tallyData.size()) * 100 << endl;
    cout << "                    Time Taken: " << timer.getElapsedTime() << " ms." << endl;
    cout << "                  Average Loss: " << totalLoss << endl;
    return {
            double(numCorrectlyClassified) / double(tallyData.size()) * 100,
            totalLoss
    };
}

vector<arma::vec> CapsuleNetwork::getErrorGradient(const vector<arma::vec> &output, int targetLabel) {
    vector<arma::vec> error = output;
    // generate the derivative of the non-linear vector activation function
    for (int i = 0; i < error.size(); i++) {
        auto l = Utils::length(error[i]);
        auto derivativeLength = (2*l) / pow(l*l + 1,2); // Note: this is the first derivative of the activation function
        error[i] = - getMarginLoss(i == targetLabel, error[i]) * derivativeLength * error[i];
    }
    return error;
}

void CapsuleNetwork::runEpoch() {
    auto& data = MNISTReader::getInstance()->trainingData;

    ProgressBar pb(data.size());
    for (int i = 0; i < data.size(); i++) {
        vector<arma::vec> output = loadImageAndGetOutput(i);
        vector<arma::vec> error = getErrorGradient(output, data[i].getLabel());
//        vector<arma::vec> imageError = getReconstructionError(output, i);

        backPropagate(error);
//        backPropagate(imageError);

        if (i%Config::batchSize == Config::batchSize-1) {
            updateWeights();
//            loadImageAndPrintOutput(i);
        }
        pb.updateProgress(i);
    }
}

void CapsuleNetwork::backPropagate(vector<arma::vec> error) {
    assert (error.size() == digitCaps.size());
    assert (error[0].size() == 16);

    vector<arma::vec> primaryCapsError(flattenTensorSize, arma::vec(8, arma::fill::zeros));
    // given the error, put this in the last layer and get the error, and give it to the Conv. net
    for (int i = 0; i < error.size(); i++) {
        vector<arma::vec> subset = digitCaps[i].backPropagate(error[i]);
        for (int j = 0; j < flattenTensorSize; j++) {
            primaryCapsError[i] += subset[i];
        }
    }
    for (auto& delta_u : primaryCapsError) {
        auto l = Utils::length(delta_u);
        auto derivativeLength = (2*l) / pow(l*l + 1, 2);
        delta_u = derivativeLength * Utils::safe_normalize(delta_u);
    }
    // translate to feature maps
    vector<FeatureMap> convError = VectorMap::toArrayOfFeatureMaps(6, 6, 16, primaryCapsError);
    // give back to the conv net here.
    primaryCaps.backPropagate(convError);
}

double CapsuleNetwork::getTotalMarginLoss(int targetLabel, const vector<arma::vec> &output) const {
    double sumOfLosses = 0.0;
    for (int i = 0; i < output.size(); i++) {
        double loss = getMarginLoss(i == targetLabel, output[i]);
        sumOfLosses += loss;
    }
    return sumOfLosses;
}

double CapsuleNetwork::getMarginLoss(bool isPresent, const arma::vec &v_k) const {
    double t_k = isPresent ? 1.0 : 0.0;
    const double m_plus = 0.9;
    const double m_minus = 0.1;
    const double lambda = 0.5;
    const double vLength = Utils::length(v_k);

    double lhs = t_k * pow(max(0.0, m_plus - vLength), 2);
    double rhs = lambda * pow(max(0.0, vLength - m_minus), 2);

    if (isPresent) {
        return lhs;
    } else {
        return rhs;
    }
}

void CapsuleNetwork::updateWeights() {
    primaryCaps.updateError();
    for (auto& cap : digitCaps) {
        cap.updateWeights();
    }
    reconstructionLayers.batchUpdate();
}

void CapsuleNetwork::train() {
    vector<pair<double, double>> history;
    for (size_t i = 0; i < Config::numEpochs; i++) {
        cout << "EPOCH ITERATION: " << i << endl;
        runEpoch();
        history.push_back(tally(false));

        // TODO file writing (and eventual reading)

        cout << endl;
        for (int j = 0; j < history.size(); j++) {
            cout << j << "," << history[j].first << "," << history[j].second << endl;
        }
    }

    cout << "DONE!" << endl;
}

vector<double> CapsuleNetwork::getErrorGradientImage(const Image& truth, const vector<double>& networkOutput) {
    vector<double> gradient(truth.size());
    for (int i = 0; i < gradient.size(); i++) {
        gradient[i] = networkOutput[i] * (1-networkOutput[i]) * (truth[i] - networkOutput[i]);
    }
    return gradient;
}