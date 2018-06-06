//
// Created by daniellopez on 4/4/18.
//

#ifndef NEURALNETS_CUCAPSULENETWORK_H
#define NEURALNETS_CUCAPSULENETWORK_H


#include <models/CUUnifiedBlob.h>
#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <armadillo>
#include <ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.h>
#include <CapsNetConfig.h>

class CUCapsuleNetwork {
public:
    CUCapsuleNetwork(const CapsNetConfig& incomingConfig);
    void forwardPropagation(int imageIndex, bool useTraining = true);
    double backPropagation(int imageIndex, bool useTraining = true);
    double getLoss();
    bool testResults(int imageIndex, bool useTraining = true);
    long double forwardAndBackPropagation(int imageIndex, bool useTraining = true);
    void runEpoch();
    pair<double, long double> tally(bool useTraining = true);
    void train();
    void updateWeights();

    void test_detailedFP();

private:
    unsigned int flattenedTensorSize;
    CUConvolutionalLayer CUPrimaryCaps;
    CUUnifiedBlob u, u_hat,
                  w, w_delta, w_velocity,
                  v,
                  b, c,
                  truth,
                  losses,
                  lengths;

    CapsNetConfig config;
};


#endif //NEURALNETS_CUCAPSULENETWORK_H
