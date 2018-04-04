//
// Created by daniellopez on 2/23/18.
//

#ifndef NEURALNETS_CAPSNET_H
#define NEURALNETS_CAPSNET_H


#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <MultilayerPerceptron/MultilayerPerceptron.h>
#include "Capsule.h"

class CapsuleNetwork {
public:
    CapsuleNetwork();
    vector<arma::vec> loadImageAndGetOutput(int imageIndex, bool useTraining = true);
    void m_threading_loadCapsuleAndGetOutput(int capsuleIndex, const vector<arma::vec> input);
    void loadImageAndPrintOutput(int imageIndex, bool useTraining = true);
    vector<arma::vec> getErrorGradient(const vector<arma::vec> &output, int targetLabel);
    vector<arma::vec> getReconstructionError(vector<arma::vec> digitCapsOutput, int imageIndex, bool useTraining = true);
    pair<double, long double> tally(bool useTraining = true);
    void backPropagate(vector<arma::vec> error);
    void runEpoch();
    void train();
    void updateWeights();

    long double getTotalMarginLoss(int targetLabel, const vector<arma::vec>& output) const;
    double getMarginLoss(bool isPresent, const arma::vec& v_k) const;
    double getMarginLossGradient(bool isPresent, const arma::vec& v_k) const;
    vector<double> getErrorGradientImage(const Image& truth, const vector<double>& networkOutput);

private:
    ConvolutionalLayer primaryCaps; // this is basicaly the primary caps in a different form
    // PrimaryCaps -> DigitCaps;
    vector<Capsule> digitCaps; // one cap for each of
    MultilayerPerceptron reconstructionLayers;
    volatile const int flattenTensorSize = 72;
};


#endif //NEURALNETS_CAPSNET_H
