//
// Created by daniellopez on 2/23/18.
//

#include <Config.h>
#include <MNISTReader.h>
#include <models/VectorMap.h>
#include <Utils.h>
#include "CapsuleNetwork/CapsuleNetwork.h"

CapsuleNetwork::CapsuleNetwork() :
        primaryCaps(Config::inputHeight, Config::inputWidth, 256, 22, 22),
        digitCaps(10, Capsule(8, 16, 1152, 1)) {

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
    const vector<FeatureMap> primaryCapsOutput = primaryCaps.getOutput();
    auto vectorMapOutput = VectorMap::toSquishedArrayOfVecs(8, primaryCapsOutput);

    // for each of the digitCaps, make them accept this as input
    vector<arma::vec> outputs(10);
    for (int i = 0; i < 10; i++) {
        outputs[i] = digitCaps[i].routingAlgorithm(vectorMapOutput, 3);
    }
    return outputs;
}

vector<arma::vec> CapsuleNetwork::getErrorGradient(int targetLabel, const vector<arma::vec> &output) {
    vector<arma::vec> error(output.size(), arma::vec(output[0].size(), arma::fill::zeros));
    error[targetLabel] = arma::vec(output[0].size(), arma::fill::ones);
}

void CapsuleNetwork::backPropagate(const vector<arma::vec> &error) {
    // given the error, put this in the last layer and get the error, and give it to the Conv. net
    vector<arma::vec> digitCapsError(error.size());
    for (int i = 0; i < error.size(); i++) {
        digitCapsError[i] = digitCaps[i].backPropagate(error[i]);
    }
    // translate to feature maps
    // give back to the conv net here.
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