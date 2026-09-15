//
// Created by daniellopez on 4/4/18.
//

#ifndef NEURALNETS_CUUNIFIEDBLOB_H
#define NEURALNETS_CUUNIFIEDBLOB_H

#include <string>
#include <CUDAClionHelper.h>

/// Flat managed-memory (`cudaMallocManaged`) buffer of `double` used as the
/// CapsNet tensor/vector/matrix substrate.
///
/// Logical shapes are recovered by index arithmetic and kernel parameters, not
/// by nested GPU container types. See docs/CUDA_ARCHITECTURE.md.
///
/// Ownership:
/// - `data` and `flagHelper` are allocated together in `allocateMemory`.
/// - Copy/assign/resize are manual; audit free paths when changing lifetime.
/// - Many `CUDA_*` wrappers call `cudaDeviceSynchronize()` after launch.
///
/// Naming: each logical primitive has a host (`name`) and device (`CUDA_name`)
/// entry. Prefer parity tests before changing either side.
class CUUnifiedBlob {
public:
    explicit CUUnifiedBlob(int pSize = 1);
    CUUnifiedBlob(const CUUnifiedBlob& other);
    ~CUUnifiedBlob();

    void copy(const CUUnifiedBlob& other);
    void resize(int newSize);
    void clear();
    void CUDA_clear();
    void print(const std::string &msg = "", int width = 1) const;
    bool operator==(const CUUnifiedBlob &other) const;
    CUUnifiedBlob& operator=(const CUUnifiedBlob &other);
    int getSize() const;
    void fillWithRandom();
    void fillSequentially();
    int hasNan() const;
    int hasInf() const;
    bool isAllZeros() const;
    bool CUDA_hasNan() const;

    double getValueAt_1D(int location) const;
    double getValueAt_2D(int x, int y , int xDim) const;
    void setValueAt_1D(int location, double incomingValue);
    void CUDA_setValueAt(int location, double incomingValue);
    void setValueAt_2D(int x, int y, int xDim, double incomingValue);
    void setValueAt_3D(int x, int y, int z, int xDim, int yDim, double incomingValue);

    /// Vote transform: for each (t,k), `output = W * input` with dims
    /// `inputDim`→`outputDim`. Buffers sized by `numClasses * tensorSize`.
    static void matrixVectorMultiplication(CUUnifiedBlob &matrix, CUUnifiedBlob &inputVector, CUUnifiedBlob &outputVector, int inputDim, int outputDim, int numClasses, int tensorSize);
    /// Device twin of `matrixVectorMultiplication`. Synchronizes before return.
    static void CUDA_matrixVectorMultiplication(CUUnifiedBlob &matrix, CUUnifiedBlob &inputVector, CUUnifiedBlob &outputVector, int inputDim, int outputDim, int numClasses, int tensorSize);

    /// Routing softmax over logits `b` into coupling coefficients `c`.
    static void vectorVectorSoftmax(CUUnifiedBlob &b, CUUnifiedBlob &c, int numClasses, int tensorSize);
    static void CUDA_vectorVectorSoftmax(CUUnifiedBlob &b, CUUnifiedBlob &c, int numClasses, int tensorSize);

    /// Weighted vote reduction into parent capsule candidates `v`.
    static void weightReduceVectors(CUUnifiedBlob &u_hat, CUUnifiedBlob &c, CUUnifiedBlob &v, int numClasses, int tensorSize, int dim);
    static void CUDA_weightReduceVectors(CUUnifiedBlob &u_hat, CUUnifiedBlob &c, CUUnifiedBlob &v, int numClasses, int tensorSize, int dim);

    /// Capsule squash nonlinearity over `numVecs` vectors of length `vecDim`.
    static void vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim);
    static void CUDA_vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim);

    /// Agreement: accumulate `u_hat · v` into routing logits `b`.
    static void vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim);
    static void CUDA_vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim);

    /// Margin-loss / error transform into class capsule state.
    static void vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim, double m_plus,
                                       double m_minus, double lambda);
    static void CUDA_vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim,
                                           double m_plus, double m_minus, double lambda);

    static void weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &c, CUUnifiedBlob &w, CUUnifiedBlob &v_error, int numClasses, int tensorSize, int innerDim, int outerDim);
    static void CUDA_weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &w, CUUnifiedBlob &delta_u_hat_error, int numClasses, int tensorSize, int innerDim, int outerDim);

    static void scaledDecompositionOfError(CUUnifiedBlob &delta_v, CUUnifiedBlob &c, CUUnifiedBlob &delta_u_hat, int numClasses, int tensorSize, int dim);
    static void CUDA_scaledDecompositionOfError(CUUnifiedBlob &delta_v, CUUnifiedBlob &c, CUUnifiedBlob &delta_u_hat, int numClasses, int tensorSize, int dim);

    static void vectorVectorMatrixProductAndSum(CUUnifiedBlob &w, CUUnifiedBlob &v_error, CUUnifiedBlob &old_u, int numClasses, int tensorSize, int innerDim, int outerDim);
    static void CUDA_vectorVectorMatrixProductAndSum(CUUnifiedBlob &w_delta, CUUnifiedBlob &delta_v, CUUnifiedBlob &old_u, int numClasses, int tensorSize, int innerDim, int outerDim);

    static void multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim);
    static void CUDA_multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim);

    /// Momentum-style elementwise weight update; mutates `w` and clears deltas.
    static void elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_delta, CUUnifiedBlob &w_velocity, int size);
    static void CUDA_elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_error, CUUnifiedBlob &w_velocity, int size);

    static void vectorSquashDerivative(CUUnifiedBlob &v, int numVecs, int vecDim, int numClasses = 1);
    static void CUDA_vectorSquashDerivative(CUUnifiedBlob &v, int numVecs, int vecDim, int numClasses = 1);

    static void convolutionalDotProduct(CUUnifiedBlob &input, CUUnifiedBlob &filter, CUUnifiedBlob &output, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);
    static void CUDA_convolutionalDotProduct(CUUnifiedBlob &input, CUUnifiedBlob &filter, CUUnifiedBlob &output, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);

    static void tensorFlatteningAndActivatedRemapping(CUUnifiedBlob &flattenedTensor, CUUnifiedBlob &tensor, int height, int width, int depth, int numClasses, int dim);
    static void CUDA_tensorFlatteningAndActivatedRemapping(CUUnifiedBlob &flattenedTensor, CUUnifiedBlob &tensor, int height, int width, int sDepth, int numClasses, int dim);

    static void reconstructingTensorFromError(CUUnifiedBlob &tensor, CUUnifiedBlob &flattenedTensor, int height, int width, int vectorDepth, int numClasses, int dim);
    static void CUDA_reconstructingTensorFromError(CUUnifiedBlob &tensor, CUUnifiedBlob &flattenedTensor, int height, int width, int vectorDepth, int numClasses, int dim);

    static void convolutionalBackPropFromError(CUUnifiedBlob &error, CUUnifiedBlob &filters, CUUnifiedBlob &delta_filters, CUUnifiedBlob &originalInput, CUUnifiedBlob &newErrorGradient, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);
    static void CUDA_convolutionalBackPropFromError(CUUnifiedBlob &error, CUUnifiedBlob &filters, CUUnifiedBlob &delta_filters, CUUnifiedBlob &originalInput, CUUnifiedBlob &newErrorGradient, int iHeight, int iWidth, int fHeight, int fWidth, int depth, int numFilters);

    static void getSquaredLength(CUUnifiedBlob &v, CUUnifiedBlob &lengths, int numClasses, int dim);
    static void CUDA_getLength(CUUnifiedBlob &v, CUUnifiedBlob &lengths, int numClasses, int dim);

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
    int *flagHelper;
};

__device__
double sharedMemoryReduce(double *shared_mem, double thread_val, int kernelIndex, int sharedMemSize);

__global__
void cu_clearOut_kernel(double *data);

__global__
void cu_hasNan_kernel(double *data, int *flag);

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
void cu_weightedTransMatrixVecMult_kernel(double *delta_u, double *w, double *delta_u_hat, int innerDim, int outerDim);

__global__
void cu_scaledDecompositionOfError(double *delta_v, double *c, double *delta_u_hat, int numClasses);

__global__
void cu_vectorVectorMatrixProductAndSum_kernel(double *w, double *delta_v, double *old_u, int numClasses, int tensorSize, int innerDim, int outerDim);

__global__
void cu_multiVectorReduction_kernel(double *u, int numClasses, int dim);

__global__
void cu_elementWiseErrorUpdate_kernel(double *w, double *w_error, double *w_velocity);

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
void cu_getLength_kernel(double *v, double *lengths);

__global__
void cu_getVectorLoss_kernel(double *v, double *truthMap, double *losses, double m_plus, double m_minus, double lambda);
#endif //NEURALNETS_CUUNIFIEDBLOB_H
