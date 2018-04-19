//
// Created by daniellopez on 4/4/18.
//

#ifndef NEURALNETS_CUCAPSULENETWORK_H
#define NEURALNETS_CUCAPSULENETWORK_H


#include <models/CUUnifiedBlob.h>
#include <ConvolutionalNetwork/ConvolutionalLayer.h>

class CUCapsuleNetwork {
public:
    CUCapsuleNetwork();
private:
    unsigned int flattenedTensorSize;
    ConvolutionalLayer primaryCaps;
    CUUnifiedBlob u, u_hat,
                  w, w_delta, w_velocity,
                  v,
                  b, c;
};


#endif //NEURALNETS_CUCAPSULENETWORK_H
