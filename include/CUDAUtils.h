//
// Created by daniellopez on 4/4/18.
//

#ifndef NEURALNETS_CUDAUTILS_H
#define NEURALNETS_CUDAUTILS_H

#include <CUDAClionHelper.h>

class CUDAUtils {
public:
    static void handleError(cudaError_t error);
};


#endif //NEURALNETS_CUDAUTILS_H
