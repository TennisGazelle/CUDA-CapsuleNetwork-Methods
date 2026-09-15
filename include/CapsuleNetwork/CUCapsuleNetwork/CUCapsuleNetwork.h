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
#include <CapsuleNetwork/CapsuleNetwork.h>

/// CUDA Capsule Network orchestration over managed-memory blobs.
///
/// Mutation contracts (see docs/KNOWN_ISSUES.md, docs/TESTING.md):
/// - `forwardPropagation` updates activation buffers; does not call weight update.
/// - `backPropagation` mutates error/delta buffers and may write into capsule state.
/// - `updateWeights` mutates `w` / convolutional filters using accumulated deltas.
/// - `tally` is the **historical** path: it backprops and updates weights even when
///   `useTraining == false`. Prefer `evaluate` in corrected mode.
/// - `evaluate` is forward-only metrics; it must not mutate learned parameters.
class CUCapsuleNetwork {
public:
    CUCapsuleNetwork(const CapsNetConfig& incomingConfig);
    void initWithSeq(CapsuleNetwork& originalWeights);

    /// Forward pass for one example into activation buffers (`u`, `u_hat`, `v`, ...).
    void forwardPropagation(int imageIndex, bool useTraining = true);

    /// Backward pass for one example; mutates deltas / error buffers.
    double backPropagation(int imageIndex, bool useTraining = true);

    double getLoss();
    bool testResults(int imageIndex, bool useTraining = true);

    /// Forward + backprop + weight update for one example (training step).
    long double forwardAndBackPropagation(int imageIndex, bool useTraining = true);

    /// Full training epoch over the training set (mutates weights).
    long double runEpoch();

    /// Historical evaluation traversal. **Mutates** weights via backprop/update
    /// even when `useTraining == false`. Kept for compatibility/characterization.
    pair<double, long double> tally(bool useTraining = true);

    /// Corrected evaluation: forward + metrics only; does not call backprop or
    /// `updateWeights`.
    pair<double, long double> evaluate(bool useTraining = true);

    /// Training loop. Corrected builds call `evaluate` for metrics; historical
    /// compatibility builds may call mutating `tally` when enabled.
    pair<double, long double> train(const string& logHeader = "");

    /// Applies accumulated deltas to transformation matrices and conv filters.
    void updateWeights();

    void test_detailedFP();
    void verificationTest();

private:
    unsigned int flattenedTensorSize;
    CUConvolutionalLayer CUPrimaryCaps;
    CUUnifiedBlob u, u_hat,
                  w, w_delta, w_velocity,
                  v,
                  b, c,
                  truth,
                  losses,
                  lengths,
                  cache;

    CapsNetConfig config;
    int totalMemoryUsage = 0;
};


#endif //NEURALNETS_CUCAPSULENETWORK_H
