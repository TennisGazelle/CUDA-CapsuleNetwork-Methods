//
// Created by daniellopez on 4/4/18.
//

#include <iostream>
#include "CUDAUtils.h"

using namespace std;

void CUDAUtils::handleError(cudaError_t error) {
    if (error != cudaSuccess) {
        cerr << "CUDA error! - " << cudaGetErrorString(error) << endl;
        exit(1);
    }
}