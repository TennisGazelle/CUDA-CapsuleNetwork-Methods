//
// Created by daniellopez on 4/4/18.
//

#ifndef NEURALNETS_CUUNIFIEDBLOB_H
#define NEURALNETS_CUUNIFIEDBLOB_H

#include <string>
#include <CUDAClionHelper.h>

class CUUnifiedBlob {
public:
    explicit CUUnifiedBlob(int pSize = 1);
    ~CUUnifiedBlob();
    void resize(int newSize);
    void clear();
    void print(const std::string& msg = "", int width = 1);
    bool operator==(const CUUnifiedBlob& other) const;

    void setValueAt_1D(int location, double incomingValue);
    void setValueAt_2D(int x, int y, int xDim, double incomingValue);
    void setValueAt_3D(int x, int y, int z, int xDim, int yDim, double incomingValue);

    static void matrixVectorMultiplication(CUUnifiedBlob& matrix,
                                           CUUnifiedBlob& inputVector,
                                           CUUnifiedBlob& outputVector,
                                           int inputDim,
                                           int outputDim);
    static void CUDA_matrixVectorMultiplication(CUUnifiedBlob &matrix,
                                                CUUnifiedBlob &inputVector,
                                                CUUnifiedBlob &outputVector,
                                                int inputDim,
                                                int outputDim,
                                                int numMultiplications);

    static void vectorVectorSoftmax(CUUnifiedBlob& b,
                                    CUUnifiedBlob& c,
                                    int numClasses,
                                    int tensorSize);
    static void CUDA_vectorVectorSoftmax(CUUnifiedBlob& b,
                                         CUUnifiedBlob& c,
                                         int numClasses,
                                         int tensorSize);

    static void weightReduceVectors(CUUnifiedBlob &u_hat,
                                    CUUnifiedBlob &c,
                                    CUUnifiedBlob &v,
                                    int numClasses,
                                    int tensorSize,
                                    int dim);
    static void CUDA_weightReduceVectors(CUUnifiedBlob &u_hat,
                                         CUUnifiedBlob &c,
                                         CUUnifiedBlob &v,
                                         int numClasses,
                                         int tensorSize,
                                         int dim);

    static void vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim);
    static void CUDA_vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim);

    static void vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim);
    static void CUDA_vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim);    
private:
    void allocateMemory();
    void deallocateMemory();
    int size;
    double* data;
    bool isGPUAllocated;
};

__global__
void cu_matrixVectorMultiplication_kernel(double* matrix, double* inputVector, double* outputVector, int inputDim, int outputDim);

__global__
void cu_vectorVectorSoftmax_kernel(double *b, double *c, int numClasses, int tensorSize);

__global__
void cu_weightReduceVector_kernel(double *u_hat, double *c, double *v, int numClasses, int tensorSize, int dim);

__global__
void cu_vectorSquash_kernel(double *v, int numVecs, int vecDim);

__global__
void cu_vectorVectorScalarProduct_kernel(double *u_hat, double *v, double *b, int numClasses, int tensorSize, int dim);
#endif //NEURALNETS_CUUNIFIEDBLOB_H
