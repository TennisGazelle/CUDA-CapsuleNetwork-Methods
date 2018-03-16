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
        primaryCaps(Config::inputHeight, Config::inputWidth, 256, 22, 22),
        digitCaps(10, Capsule(8, 16, 1152, 1)),
        reconstructionLayers(10*16, 28*28, {512}) {
    // TODO extrapolate this for varying inputs and outputs
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
        workers[i] = thread(&CapsuleNetwork::loadCapsuleAndGetOutput, this, i, vectorMapOutput);
    }

    // allocate a thread for each one of these
    for (int i = 0; i < digitCaps.size(); i++) {
        workers[i].join();
        outputs[i] = digitCaps[i].getOutput();
    }
    return outputs;
}

void CapsuleNetwork::loadCapsuleAndGetOutput(int capsuleIndex, const vector<arma::vec> input) {
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
            digitCapsOutput[i] = arma::vec(digitCapsOutput[0].size(), arma::fill::zeros);
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
    const vector<FeatureMap> primaryCapsOutput = primaryCaps.getOutput();
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

//    cout << endl;
//    auto reconstructionImage = reconstructionLayers.loadInputAndGetOutput(Utils::getAsOneDim(digitCapsOutput));
//    cout.precision(3);
//    cout << fixed;
//    for (int r = 0; r < 28; r++) {
//        for (int c = 0; c < 28; c++) {
//            cout << reconstructionImage[r*28 + c] << " ";
//        }
//        cout << endl;
//    }
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

vector<arma::vec> CapsuleNetwork::getErrorGradient(const vector<arma::vec> &output, int targetLabel) {
    const static double lambda = 0.5, m_plus = 0.9, m_minus = (1 - m_plus);
    vector<arma::vec> error = output;
    for (int i = 0; i < error.size(); i++) {
        auto l = Utils::length(error[i]);

        if (i == targetLabel) {
            // T_k == 1
            // is the length of this vector's negative bigger than 0?
            error[i] *= - pow(max(0.0, m_plus - l), 2);
        } else {
            // T_k == 0
            // is the length of this vector bigger than 0?
            error[i] *= - lambda * pow(max(0.0, l - m_minus), 2);
        }
    }

//    vector<arma::vec> error = output;
//    error[targetLabel] = normalise(output[targetLabel]);
//    error[targetLabel] = normalise(arma::vec(output[0].size(), arma::fill::ones));
    return error;
}

void CapsuleNetwork::runEpoch() {
    auto& data = MNISTReader::getInstance()->testingData;

    ProgressBar pb(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        vector<arma::vec> output = loadImageAndGetOutput(i);
        vector<arma::vec> error = getErrorGradient(output, data[i].getLabel());
        vector<arma::vec> imageError = getReconstructionError(output, i);

        backPropagate(error);
        backPropagate(imageError);

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

    vector<arma::vec> primaryCapsError(1152, arma::vec(8, arma::fill::zeros));
    // given the error, put this in the last layer and get the error, and give it to the Conv. net
    for (int i = 0; i < error.size(); i++) {
        vector<arma::vec> subset = digitCaps[i].backPropagate(error[i]);
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
    reconstructionLayers.batchUpdate();
}

void CapsuleNetwork::train() {
    vector<double> history;
    for (size_t i = 0; i < Config::numEpochs; i++) {
        cout << "EPOCH ITERATION: " << i << endl;
        runEpoch();
        history.push_back(tally(false));

        // TODO file writing (and eventual reading)

        cout << endl;
        for (int i = 0; i < history.size(); i++) {
        	cout << i << "," << history[i] << endl;
        }
    }

    cout << "DONE!" << endl;
}

vector<double> CapsuleNetwork::getErrorGradientImage(const Image& truth, const vector<double>& networkOutput) {
    vector<double> gradient(truth.size());
    for (int i = 0; i < gradient.size(); i++) {
//        gradient[i] = truth[i] - networkOutput[i];
        gradient[i] = networkOutput[i] * (1-networkOutput[i]) * (truth[i] - networkOutput[i]);
        gradient[i] = pow(networkOutput[i] - truth[i], 2);
    }
    return gradient;
}