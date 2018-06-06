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
    CUUnifiedBlob(const CUUnifiedBlob& other);
    ~CUUnifiedBlob();

    void copy(const CUUnifiedBlob& other);
    void resize(int newSize);
    void clear();
    void CUDA_clear();
    void print(const std::string &msg = "", int width = 1);
    bool operator==(const CUUnifiedBlob &other) const;
    int getSize() const;
    void fillWithRandom();

    double getValueAt_1D(int location) const;
    double getValueAt_2D(int x, int y , int xDim) const;
    void setValueAt_1D(int location, double incomingValue);
    void CUDA_setValueAt(int location, double incomingValue);
    void setValueAt_2D(int x, int y, int xDim, double incomingValue);
    void setValueAt_3D(int x, int y, int z, int xDim, int yDim, double incomingValue);

    static void matrixVectorMultiplication(CUUnifiedBlob &matrix, CUUnifiedBlob &inputVector, CUUnifiedBlob &outputVector, int inputDim, int outputDim, int numClasses, int tensorSize);
    static void CUDA_matrixVectorMultiplication(CUUnifiedBlob &matrix, CUUnifiedBlob &inputVector, CUUnifiedBlob &outputVector, int inputDim, int outputDim, int numClasses, int tensorSize);

    static void vectorVectorSoftmax(CUUnifiedBlob &b, CUUnifiedBlob &c, int numClasses, int tensorSize);
    static void CUDA_vectorVectorSoftmax(CUUnifiedBlob &b, CUUnifiedBlob &c, int numClasses, int tensorSize);

    static void weightReduceVectors(CUUnifiedBlob &u_hat, CUUnifiedBlob &c, CUUnifiedBlob &v, int numClasses, int tensorSize, int dim);
    static void CUDA_weightReduceVectors(CUUnifiedBlob &u_hat, CUUnifiedBlob &c, CUUnifiedBlob &v, int numClasses, int tensorSize, int dim);

    static void vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim);
    static void CUDA_vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim);

    static void vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim);
    static void CUDA_vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim);

    static void vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim, double m_plus,
                                       double m_minus, double lambda);
    static void CUDA_vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim,
                                           double m_plus, double m_minus, double lambda);

    static void weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &c, CUUnifiedBlob &w, CUUnifiedBlob &v_error, int numClasses, int tensorSize, int innerDim, int outerDim);
    static void CUDA_weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &w, CUUnifiedBlob &v_error, int numClasses, int tensorSize, int innerDim, int outerDim);

    static void scaledDecompositionOfError(CUUnifiedBlob &delta_v, CUUnifiedBlob &c, CUUnifiedBlob &delta_u_hat, int numClasses, int tensorSize, int dim);
    static void CUDA_scaledDecompositionOfError(CUUnifiedBlob &delta_v, CUUnifiedBlob &c, CUUnifiedBlob &delta_u_hat, int numClasses, int tensorSize, int dim);

    static void vectorVectorMatrixProductAndSum(CUUnifiedBlob &w, CUUnifiedBlob &v_error, CUUnifiedBlob &old_u, int numClasses, int tensorSize, int innerDim, int outerDim);
    static void CUDA_vectorVectorMatrixProductAndSum(CUUnifiedBlob &w, CUUnifiedBlob &v_error, CUUnifiedBlob &old_u, int numClasses, int tensorSize, int innerDim, int outerDim);

    static void multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim);
    static void CUDA_multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim);

    static void elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_delta, CUUnifiedBlob &w_velocity, int size);
    static void CUDA_elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_error, CUUnifiedBlob &w_velocity, int size);

    static void vectorSquashDerivative(CUUnifiedBlob &v, int numVecs, int vecDim, int numClasses = 1);
    static void CUDA_vectorSquashDerivative(CUUnifiedBlob &v, int numVecs, int vecDim, int numClasses = 1);

    static void convolutionalDotProduct(CUUnifiedBlob &input, CUUnifiedBlob &filter, CUUnifiedBlob &output, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);
    static void CUDA_convolutionalDotProduct(CUUnifiedBlob &input, CUUnifiedBlob &filter, CUUnifiedBlob &output, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);

    static void tensorFlatteningAndActivatedRemapping(CUUnifiedBlob &flattenedTensor, CUUnifiedBlob &tensor, int height, int width, int depth, int numClasses, int dim);
    static void CUDA_tensorFlatteningAndActivatedRemapping(CUUnifiedBlob &flattenedTensor, CUUnifiedBlob &tensor, int height, int width, int sDepth, int numClasses, int dim);

    static void reconstructingTensorFromError(CUUnifiedBlob &tensor, CUUnifiedBlob &flattenedTensor, int height, int width, int depth, int numClasses, int dim);
    static void CUDA_reconstructingTensorFromError(CUUnifiedBlob &tensor, CUUnifiedBlob &flattenedTensor, int height, int width, int depth, int numClasses, int dim);

    static void convolutionalBackPropFromError(CUUnifiedBlob &error, CUUnifiedBlob &filters, CUUnifiedBlob &delta_filters, CUUnifiedBlob &originalInput, CUUnifiedBlob &newErrorGradient, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);
    static void CUDA_convolutionalBackPropFromError(CUUnifiedBlob &error, CUUnifiedBlob &filters, CUUnifiedBlob &delta_filters, CUUnifiedBlob &originalInput, CUUnifiedBlob &newErrorGradient, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);

    static void getSquaredLength(CUUnifiedBlob &v, CUUnifiedBlob &lengths, int numClasses, int dim);
    static void CUDA_getSquaredLength(CUUnifiedBlob &v, CUUnifiedBlob &lengths, int numClasses, int dim);

    static void getVectorLoss(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, CUUnifiedBlob &losses, int numClasses, int dim,
                                  double m_plus, double m_minus, double lambda);
    static void CUDA_getVectorLoss(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, CUUnifiedBlob &losses, int numClasses,
                                       int dim, double m_plus, double m_minus, double lambda);
private:
    void allocateMemory();
    void deallocateMemory();

    int size;
    double *data;
    bool isGPUAllocated;
};

// https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error/37569519
//#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//#define ATOMIC_ADD_DEFINITION_REQUIRED
//__device__
//double atomicAdd(double *addr, double val);
//#endif

__device__
double sharedMemoryReduce(double *shared_mem, double thread_val, int kernelIndex);

__global__
void cu_clearOut_kernel(double *data);

__global__
void cu_singleElementSetting_kernel(double *data, int location, double incomingValue);

__global__
void cu_matrixVectorMultiplication_kernel(double *matrix, double *inputVector, double *outputVector, int inputDim,
                                          int outputDim);

__global__
void cu_vectorVectorSoftmax_kernel(double *b, double *c, int numClasses, int tensorSize);

__global__
void cu_weightReduceVector_kernel(double *u_hat, double *c, double *v, int numClasses, int tensorSize, int dim);

__global__
void cu_vectorSquash_kernel(double *v, int numVecs, int vecDim);

__global__
void cu_vectorVectorScalarProduct_kernel(double *u_hat, double *v, double *b, int numClasses, int tensorSize, int dim);

__global__
void cu_vectorLossFunction_kernel(double *v, double *truthMap, double m_plus, double m_minus, double lambda);

__global__
void cu_weightedTransMatrixVecMult_kernel(double *delta_u, double *w, double *v_error, int innerDim, int outerDim);

__global__
void cu_scaledDecompositionOfError(double *delta_v, double *c, double *delta_u_hat, int numClasses);

__global__
void cu_vectorVectorMatrixProductAndSum_kernel(double *w, double *v_error, double *old_u, int numClasses, int tensorSize, int innerDim, int outerDim);

__global__
void cu_multiVectorReduction_kernel(double *u, int numClasses, int dim);

__global__
void cu_elementWiseErrorUpdate_kernel(double *w, double *w_error, double *w_velocity);

__global__
void cu_errorUpdate_kernel(double *w);

__global__
void cu_vectorSquashDerivative_kernel(double *v, int numClasses);

__global__
void cu_convolutionalDotProduct_kernel(double *input, double *filter, double *output, int iHeight, int iWidth);

__global__
void cu_tensorFlatteningAndActivatedRemapping_kernel(double *flattenedTensor, double *tensor, int numClasses);

__global__
void cu_reconstructingTensorFromError_kernel(double *tensor, double *flattenedTensor, int numClasses);

__global__
void cu_convolutionalBackPropFromError_kernel(double *error, double *filters, double *delta_filters, double *originalInput, double *newErrorGradient, int iHeight, int iWidth);

__global__
void cu_getSquaredLength_kernel(double *v, double *lengths);

__global__
void cu_getVectorLoss_kernel(double *v, double *truthMap, double *losses, double m_plus, double m_minus, double lambda);
#endif //NEURALNETS_CUUNIFIEDBLOB_H
