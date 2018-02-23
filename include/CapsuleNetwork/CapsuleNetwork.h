//
// Created by daniellopez on 2/23/18.
//

#ifndef NEURALNETS_CAPSNET_H
#define NEURALNETS_CAPSNET_H


#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include "Capsule.h"

class CapsuleNetwork {
public:
    CapsuleNetwork();
    vector<arma::vec> loadImageAndGetOutput(int imageIndex, bool useTraining = true);

private:
    ConvolutionalLayer primaryCaps; // this is basicaly the primary caps in a different form
    // PrimaryCaps -> DigitCaps;
    vector<Capsule> digitCaps; // one cap for each of
};


#endif //NEURALNETS_CAPSNET_H
