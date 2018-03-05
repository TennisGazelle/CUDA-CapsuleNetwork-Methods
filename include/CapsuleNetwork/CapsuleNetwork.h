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
    void loadCapsuleAndGetOutput(int capsuleIndex, const vector<arma::vec> input);
    Image loadImageAndGetReconstruction(int imageIndex, bool useTraining = true);
    void loadImageAndPrintOutput(int imageIndex, bool useTraining = true);
    double tally(bool useTraining = true);
    vector<arma::vec> getErrorGradient(int targetLabel, const vector<arma::vec>& output);
    void backPropagate(vector<arma::vec> error);
    void runEpoch();
    void train();
    void updateWeights();

    double getTotalMarginLoss(int targetLabel, const vector<arma::vec>& output) const;
    double getMarginLoss(bool isPresent, const arma::vec& v_k) const;
    vector<double> getErrorGradientImage(const Image& truth, const Image& networkOutput);

private:
    ConvolutionalLayer primaryCaps; // this is basicaly the primary caps in a different form
    // PrimaryCaps -> DigitCaps;
    vector<Capsule> digitCaps; // one cap for each of
    MultilayerPerceptron reconstructionLayers;
};


#endif //NEURALNETS_CAPSNET_H
