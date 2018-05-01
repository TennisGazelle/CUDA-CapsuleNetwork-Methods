//
// Created by daniellopez on 4/4/18.
//

#ifndef NEURALNETS_CUCAPSULENETWORK_H
#define NEURALNETS_CUCAPSULENETWORK_H


#include <models/CUUnifiedBlob.h>
#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <armadillo>
#include <ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.h>

class CUCapsuleNetwork {
public:
    CUCapsuleNetwork();
    void forwardPropagation(int imageIndex, bool useTraining = true);
    void backPropagation(int imageIndex, bool useTraining = true);
    bool testResults(int imageIndex, bool useTraining = true);
    long double forwardAndBackPropagation(int imageIndex, bool useTraining = true);
    void runEpoch();
    void updateWeights();

private:
    void to1DSquishedArrayOfVecs(size_t vectorDim, vector<FeatureMap> inputMaps, CUUnifiedBlob &output, int numClasses) const;
    unsigned int flattenedTensorSize;
    ConvolutionalLayer primaryCaps;
    CUConvolutionalLayer other_primaryCaps;
    CUUnifiedBlob u, u_hat,
                  w, w_delta, w_velocity,
                  v,
                  b, c,
                  truth;
};


#endif //NEURALNETS_CUCAPSULENETWORK_H
