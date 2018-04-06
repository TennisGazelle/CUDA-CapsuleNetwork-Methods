//
// Created by daniellopez on 4/4/18.
//

#include <cassert>
#include <CUDAUtils.h>
#include <iostream>
#include "models/CUUnifiedBlob.h"

using namespace std;

CUUnifiedBlob::CUUnifiedBlob(int pSize) : size(pSize), data(nullptr), isGPUAllocated(false) {
    assert (pSize > 0);
    allocateMemory();
}

CUUnifiedBlob::~CUUnifiedBlob() {
    if (isGPUAllocated) {
        deallocateMemory();
    }
}

void CUUnifiedBlob::allocateMemory() {
    assert(!isGPUAllocated);
    auto error = cudaMallocManaged((void**)&data, size * sizeof(double), cudaMemAttachGlobal);
    CUDAUtils::handleError(error);
    isGPUAllocated = true;
}

void CUUnifiedBlob::deallocateMemory() {
    assert(isGPUAllocated);
    cudaFree(data);
    data = nullptr;
    isGPUAllocated = false;
}

void CUUnifiedBlob::clear() {
    for (int i = 0; i < size; i++) {
        data[i] = 0.0;
    }
}

void CUUnifiedBlob::resize(int newSize) {
    deallocateMemory();
    size = newSize;
    allocateMemory();
}

void CUUnifiedBlob::print(const string& msg) {
    if (!msg.empty()) {
        cout << msg << endl;
    }
    int bufferSize = min(size, 100);
    for (int i = 0; i < bufferSize; i++) {
        cout << data[i] << "\t" << endl;
    }
    cout << endl;
}

void CUUnifiedBlob::setValueAt_1D(int location, double incomingValue) {
    assert(0 <= location && location < size);
    data[location] = incomingValue;
}

void CUUnifiedBlob::setValueAt_2D(int x, int y, int xDim, double incomingValue) {
    // where is the location?
    int location = x * xDim;
    location += y;

    setValueAt_1D(location, incomingValue);
}

void CUUnifiedBlob::setValueAt_3D(int x, int y, int z, int xDim, int yDim, double incomingValue) {
    int location = z * xDim * yDim;
    location += x * xDim;
    location += y;

    setValueAt_1D(location, incomingValue);
}

void CUUnifiedBlob::matrixVectorMultiplication(CUUnifiedBlob &matrix, CUUnifiedBlob &inputVector,
                                               CUUnifiedBlob &outputVector, int inputDim, int outputDim) {
    assert(matrix.size == inputDim*outputDim);
    assert(inputVector.size == inputDim);
    assert(outputVector.size == outputDim);

    for (int i = 0; i < outputDim; i++) {
        for (int j = 0; j < inputDim; j++) {
            cout << "output at += : " << i << " matrix:[" << i  << ", " << j << "]" << endl;
            outputVector.data[i] += inputVector.data[j] * matrix.data[i*inputDim + j];
        }
    }
}

void CUUnifiedBlob::CUDA_matrixVectorMultiplication(CUUnifiedBlob &matrix,
                                                    CUUnifiedBlob &inputVector,
                                                    CUUnifiedBlob &outputVector,
                                                    int inputDim,
                                                    int outputDim) {

    cu_matrixVectorMultiplication_helper<<<1, outputDim>>>(matrix.data,
                                                    inputVector.data,
                                                    outputVector.data,
                                                    inputDim,
                                                    outputDim);
}

__global__
void cu_matrixVectorMultiplication_helper(double *matrix,
                                          double *inputVector,
                                          double *outputVector,
                                          int inputDim,
                                          int outputDim) {
    int r = threadIdx.x;
    for (int c = 0; c < inputDim; c++) {
        outputVector[r] += matrix[r*inputDim+c] * inputVector[c];
    }
}