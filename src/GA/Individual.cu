// CUDA-dependent fitness evaluation for GA individuals.

#include "GA/Individual.h"

#include <CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h>
#include <GA/CapsNetDAO.h>

#include <iostream>

void Individual::evaluate() {
    decodeChromosome();

    if (CapsNetDAO::getInstance()->isInDatabase(*this)) {
        CapsNetDAO::getInstance()->getFromDatabase(*this);
        return;
    }

    constructNetworkAndEvaluate();
    CapsNetDAO::getInstance()->addToDatabase(*this);
}

void Individual::constructNetworkAndEvaluate() {
    cout << "evaluating chromosome: " << to_string() << endl;
    fullPrint();

    CUCapsuleNetwork network(capsNetConfig);

    auto fitness = network.train(to_string());
    accuracy_100 = fitness.first;
    loss_100 = fitness.second;

    fitness = network.train(to_string());
    accuracy_300 = fitness.first;
    loss_300 = fitness.second;
}

void Individual::fakeEvaluate() {
    // Explicit synthetic fitness path retained only for algorithm-level GA
    // experiments. Unlike the historical implementation, this method does not
    // construct or train a CUDA network and is never called by evaluate().
    decodeChromosome();

    loss_100 = static_cast<double>(capsNetConfig.cnInnerDim) *
               static_cast<double>(capsNetConfig.cnOuterDim) *
               static_cast<double>(capsNetConfig.cnNumTensorChannels);
    loss_100 += capsNetConfig.lambda * capsNetConfig.m_minus * capsNetConfig.m_plus;
    loss_300 = loss_100 * capsNetConfig.lambda;

    accuracy_100 = capsNetConfig.lambda * capsNetConfig.m_plus /
                   static_cast<double>(capsNetConfig.cnInnerDim *
                                       capsNetConfig.cnOuterDim *
                                       capsNetConfig.cnNumTensorChannels);
    accuracy_300 = accuracy_100 * capsNetConfig.cnNumTensorChannels;
}
