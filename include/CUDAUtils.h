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

__device__
int getGlobalIdx_1D_1D();
__device__
int getGlobalIdx_1D_2D();
__device__
int getGlobalIdx_1D_3D();

__device__
int getGlobalIdx_2D_1D();
__device__
int getGlobalIdx_2D_2D();
__device__
int getGlobalIdx_2D_3D();

__device__
int getGlobalIdx_3D_1D();
__device__
int getGlobalIdx_3D_2D();
__device__
int getGlobalIdx_3D_3D();

#endif //NEURALNETS_CUDAUTILS_H
