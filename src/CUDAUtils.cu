//
// Created by daniellopez on 4/4/18.
//

#include <iostream>
#include "CUDAUtils.h"

using namespace std;

void CUDAUtils::handleError(cudaError_t error, const std::string& debugMsg) {
    if (error != cudaSuccess) {
        cerr << "CUDA error! - " << cudaGetErrorString(error) << endl;
        if (!debugMsg.empty()) {
        	cerr << "Debug Message: " << debugMsg << endl;
        }
        exit(1);
    }
}

void CUDAUtils::checkForError(const std::string& locationMsg) {
	handleError(cudaPeekAtLastError(), "Location: " + locationMsg);
}

__device__
int getGlobalIdx_1D_1D() {
    return blockIdx.x *blockDim.x + threadIdx.x;
}
__device__
int getGlobalIdx_1D_2D() {
    return blockIdx.x * blockDim.x * blockDim.y
           + threadIdx.y * blockDim.x + threadIdx.x;
}
__device__
int getGlobalIdx_1D_3D() {
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
           + threadIdx.z * blockDim.y * blockDim.x
           + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__
int getGlobalIdx_2D_1D() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}
__device__
int getGlobalIdx_2D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}
__device__
int getGlobalIdx_2D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__
int getGlobalIdx_3D_1D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}
__device__
int getGlobalIdx_3D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}
__device__
int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}