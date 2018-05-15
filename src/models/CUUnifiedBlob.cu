//
// Created by daniellopez on 4/4/18.
//

#include <cassert>
#include <CUDAUtils.h>
#include <iostream>
#include <cmath>
#include <Utils.h>
#include <math_functions.h>
#include "models/CUUnifiedBlob.h"
#include "CUDAUtils.h"

CUUnifiedBlob::CUUnifiedBlob(int pSize) : size(pSize), data(nullptr), isGPUAllocated(false) {
    assert (pSize > 0);
    allocateMemory();
}

CUUnifiedBlob::CUUnifiedBlob(const CUUnifiedBlob &other) {
    copy(other);
}

CUUnifiedBlob::~CUUnifiedBlob() {
    if (isGPUAllocated) {
        deallocateMemory();
    }
}

void CUUnifiedBlob::allocateMemory() {
    assert(!isGPUAllocated);
    auto error = cudaMallocManaged((void **) &data, size * sizeof(double));
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

void CUUnifiedBlob::CUDA_clear() {
    cu_clearOut_kernel << < size, 1 >> > (data);
}

void CUUnifiedBlob::resize(int newSize) {
    deallocateMemory();
    size = newSize;
    allocateMemory();
}

void CUUnifiedBlob::copy(const CUUnifiedBlob &other) {
    if (other.size != size) {
        resize(other.size);
    }
    for (int i = 0; i < size; i++) {
        data[i] = other.data[i];
    }
}

void CUUnifiedBlob::print(const std::string &msg, int width) {
    cudaDeviceSynchronize();
    if (!msg.empty()) {
        std::cout << msg << std::endl;
    }
    int bufferSize = std::min(size, 1000);
    for (int i = 0; i < bufferSize; i++) {
        std::cout << data[i] << "\t";
        if (((i + 1) % width) == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
//    for (int i = 0; i < size; i++) {
//        if (data[i] == 0.0) {
//            std::cout << "zero at location: " << i << "(" << i/width << "," << i % width << ")" << std::endl;
//        }
//    }
}

bool CUUnifiedBlob::operator==(const CUUnifiedBlob &other) const {
    if (this == &other) {
        return true;
    }

    if (size != other.size) {
        std::cout << "bad sizes" << std::endl;
        return false;
    }

    for (int i = 0; i < size; i++) {
        if (data[i] != other.data[i]) {
            sleep(1);
            if (data[i] != other.data[i]) {
                std::cout << "they didn't match at: " << i << std::endl;
                std::cout << "this: " << data[i] << " other: " << other.data[i] << std::endl;
                return false;
            }
        }
    }

    return true;
}

int CUUnifiedBlob::getSize() const {
    return size;
}

void CUUnifiedBlob::fillWithRandom() {
    for (int i = 0; i < size; i++) {
        data[i] = Utils::getWeightRand(1);
    }
}

double CUUnifiedBlob::getValueAt_1D(int location) const {
    assert(0 <= location && location < size);
    return data[location];
}

double CUUnifiedBlob::getValueAt_2D(int x, int y, int xDim) const {
    int location = x*xDim + y;
    return getValueAt_1D(location);
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

void CUUnifiedBlob::matrixVectorMultiplication(CUUnifiedBlob &matrix, CUUnifiedBlob &inputVector, CUUnifiedBlob &outputVector,
                                               int inputDim, int outputDim, int numClasses, int tensorSize) {
    for (int t = 0; t < tensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {

            int elementIndex = t*numClasses + k;
            int inputIndex = elementIndex*inputDim;
            int outputIndex = elementIndex*outputDim;
            int matrixIndex = elementIndex*inputDim*outputDim;

            for (int i = 0; i < outputDim; i++) {
                for (int j = 0; j < inputDim; j++) {
                    double cellValue = inputVector.data[j + inputIndex] * matrix.data[(i * inputDim + j) + matrixIndex];
                    if (!isnan(cellValue)) {
                        outputVector.data[i + outputIndex] += cellValue;
                    }
                }
            }
        }
    }
}

void CUUnifiedBlob::vectorVectorSoftmax(CUUnifiedBlob &b, CUUnifiedBlob &c,
                                        int numClasses, int tensorSize) {
    for (int k = 0; k < numClasses; k++) {
        double sum_b_exps = 0.0;
        for (int t = 0; t < tensorSize; t++) {
            double exp_val = exp(b.data[t * numClasses + k]);
            if (isnan(exp_val)) {
                exp_val = 0;
            }
            if (!isinf(sum_b_exps + exp_val)) {
                sum_b_exps += exp_val;
            }
        }

        // then go through the c's and set accordingly
        for (int t = 0; t < tensorSize; t++) {
            double exp_val = exp(b.data[t * numClasses + k]);
            if (isnan(exp_val)) {
                exp_val = 0;
            }
            c.data[t * numClasses + k] = exp_val / sum_b_exps;
        }
    }
}

void CUUnifiedBlob::weightReduceVectors(CUUnifiedBlob &u_hat, CUUnifiedBlob &c, CUUnifiedBlob &v, int numClasses,
                                        int tensorSize, int dim) {
    for (int k = 0; k < numClasses; k++) {
        for (int t = 0; t < tensorSize; t++) {
            int u_hat_index = t * numClasses * dim + k * dim;

            for (int i = u_hat_index; i < u_hat_index + dim; i++) {
                v.data[i % (numClasses * dim)] += u_hat.data[i] * c.data[t * numClasses + k];
            }
        }
    }
}

void CUUnifiedBlob::vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim) {
    for (int v_index = 0; v_index < numVecs * vecDim; v_index += vecDim) {
        double sum_squares = 0;
        for (int i = 0; i < vecDim; i++) {
            sum_squares += std::pow(v.data[v_index + i], 2);
        }
        double squashFactor = sum_squares / (1.0 + sum_squares);
        sum_squares = sqrt(sum_squares);
        for (int i = 0; i < vecDim; i++) {
            v.data[v_index + i] *= squashFactor / sum_squares;
        }
    }
}

void CUUnifiedBlob::vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses,
                                              int tensorSize, int dim) {
    for (int k = 0; k < numClasses; k++) {
        int v_index = k * dim;
        for (int t = 0; t < tensorSize; t++) {
            int u_hat_index = t * numClasses * dim + k * dim;
            int b_index = t * numClasses + k;

            for (int i = 0; i < dim; i++) {
                double b_inc = u_hat.data[u_hat_index + i] * v.data[v_index + i];
                if (!isnan(b_inc)) {
                    b.data[b_index] += b_inc;
                }
            }
        }
    }
}

void CUUnifiedBlob::vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim) {
    const double m_plus = 0.9;
    const double m_minus = 0.1;
    const double lambda = 0.5;

    for (int k = 0; k < numClasses; k++) {
        double sumOfSquaredValues = 0.0;
        for (int i = 0; i < dim; i++) {
            sumOfSquaredValues += std::pow(v.data[k * dim + i], 2);
        }
        double vec_length = sqrt(sumOfSquaredValues + 1e-4);

        double activationFactor = 2 * vec_length / pow((vec_length * vec_length) + 1, 2);

        double errorGradient;
        if (vec_length < m_plus) {
            if (vec_length <= m_minus) {
                errorGradient = -2 * truthMap.data[k] * (m_plus - vec_length);
            } else {
                errorGradient = 2 * ((lambda * (truthMap.data[k] - 1) * (m_minus - vec_length)) +
                                     truthMap.data[k] * (vec_length - m_plus));
            }
        } else {
            errorGradient = 2 * lambda * (truthMap.data[k] - 1) * (m_minus - vec_length);
        }

        double rawMarginLoss;
        if (truthMap.data[k]) {
            rawMarginLoss = pow(std::max(0.0, m_plus - vec_length), 2);
        } else {
            rawMarginLoss = lambda * pow(std::max(0.0, vec_length - m_minus), 2);
        }

        double resizingFactor = activationFactor * errorGradient * rawMarginLoss / vec_length;
        for (int i = 0; i < dim; i++) {
            v.data[k * dim + i] *= resizingFactor;
        }
    }
}

void CUUnifiedBlob::weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &c, CUUnifiedBlob &w,
                                               CUUnifiedBlob &v_error, int numClasses,
                                               int tensorSize, int innerDim, int outerDim) {
    for (int t = 0; t < tensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            int c_index = t * numClasses + k;
            int w_index = c_index * innerDim * outerDim;
            int v_index = k * outerDim;
            int u_index = c_index * innerDim;
            // think transposed matrix
            for (int col = 0; col < innerDim; col++) {
                double u_value = 0.0;
                for (int row = 0; row < outerDim; row++) {
                    u_value += w.data[row * innerDim + col + w_index] * v_error.data[row + v_index];
                }
                delta_u.data[col + u_index] = u_value * c.data[c_index];
            }
        }
    }
}

void CUUnifiedBlob::vectorVectorMatrixProductAndSum(CUUnifiedBlob &w, CUUnifiedBlob &v_error, CUUnifiedBlob &old_u,
                                                    int numClasses, int tensorSize, int innerDim, int outerDim) {
    for (int t = 0; t < tensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            int element_index = t * numClasses + k;
            int w_index = element_index * innerDim * outerDim;
            int v_index = k * outerDim;
            int u_index = element_index * innerDim;

            for (int row = 0; row < outerDim; row++) {
                for (int col = 0; col < innerDim; col++) {
                    w.data[row * innerDim + col + w_index] += v_error.data[row + v_index] * old_u.data[col + u_index];
                }
            }
        }
    }
}

void CUUnifiedBlob::multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim) {
    for (int t = 0; t < tensorSize; t++) {
        for (int d = 0; d < dim; d++) {
            for (int k = 1; k < numClasses; k++) {
                int element_index = t * numClasses + k;
                u.data[t * numClasses * dim + d] += u.data[element_index * dim + d];
                u.data[element_index * dim + d] = 0;
            }
        }
    }
}

void CUUnifiedBlob::elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_error, CUUnifiedBlob &w_velocity, int size) {
    for (int i = 0; i < size; i++) {
        w_velocity.data[i] = w_velocity.data[i]*0.9 + w_error.data[i]*0.1;
        w.data[i] += w_velocity.data[i];
        w_error.data[i] = 0;
    }
}

void CUUnifiedBlob::vectorSquashDerivative(CUUnifiedBlob &v, int numVecs, int vecDim, int numClasses) {
    for (int v_index = 0; v_index < numVecs * vecDim * numClasses; v_index += vecDim*numClasses) {
        double sum_squares = 0;
        for (int i = 0; i < vecDim; i++) {
            sum_squares += pow(v.data[v_index + i], 2);
        }
        sum_squares = sqrt(sum_squares);
        double squashFactor = (2 * sum_squares) / pow(sum_squares * sum_squares + 1, 2);

        for (int i = 0; i < vecDim; i++) {
            v.data[v_index + i] *= squashFactor / sum_squares;
        }
    }
}

void CUUnifiedBlob::convolutionalDotProduct(CUUnifiedBlob &input, CUUnifiedBlob &filter, CUUnifiedBlob &output,
                                            int iHeight, int iWidth, int fHeight, int fWidth, int depth,
                                            int numFilters) {
    int outputHeight = iHeight - fHeight;
    int outputWidth = iWidth - fWidth;

    for (int f = 0; f < numFilters; f++) {
        for (int r = 0; r < outputHeight; r++) {
            for (int c = 0; c < outputWidth; c++) {

                double sum = 0.0;
                for (int ch = 0; ch < depth; ch++) {
                    for (int fr = 0; fr < fHeight; fr++) {
                        for (int fc = 0; fc < fWidth; fc++) {
                            int filterIndex = f * depth * fHeight * fWidth + ch * fHeight * fWidth + fr * fWidth + fc;
                            int inputIndex = ch * iHeight * iWidth + (r + fr) * iWidth + (c + fc);
                            sum += input.data[inputIndex] * filter.data[filterIndex];
                        }
                    }
                }
                sum /= depth * fHeight * fWidth;

                int outputIndex = f * outputHeight * outputWidth + r * outputWidth + c;
                output.setValueAt_1D(outputIndex, sum);
            }
        }
    }
}

void CUUnifiedBlob::tensorFlatteningAndActivatedRemapping(CUUnifiedBlob &flattenedTensor, CUUnifiedBlob &tensor,
                                                          int height, int width, int depth, int numClasses, int dim) {
    // go through each vector (filterDepth first, then width, then height)
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            for (int d = 0; d < depth; d++) {
                int vector_index = r * depth * width + c * depth + d;

                arma::vec v(dim);
                for (int dimIndex = 0; dimIndex < dim; dimIndex++) {
                    int tensor_index = (d * dim + dimIndex) * height * width + r * width + c;
                    v[dimIndex] = tensor.data[tensor_index];
                }
                // squash it
                v = Utils::squish(v);

                // save it in leftmost column
                for (int dimIndex = 0; dimIndex < dim; dimIndex++) {
                    for (int k = 0; k < numClasses; k++) {
                        flattenedTensor.setValueAt_2D(vector_index, dimIndex + (k * dim), numClasses * dim,
                                                      v[dimIndex]);
                    }
                }
            }
        }
    }
}

void CUUnifiedBlob::reconstructingTensorFromError(CUUnifiedBlob &tensor, CUUnifiedBlob &flattenedTensor, int height,
                                                  int width, int depth, int numClasses, int dim) {
    // go through each of the vector (filterDepth first, then width, then height)
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            for (int d = 0; d < depth; d++) {
                // collect error from the thing
                int vector_index = r * depth * width + c * width + d;

                for (int dimIndex = 0; dimIndex < dim; dimIndex++) {
                    int tensor_index = (d * dim + dimIndex) * height * width + r * width + c;
                    int error_index = (vector_index) * (numClasses * dim) + dimIndex;
                    tensor.data[tensor_index] = flattenedTensor.data[error_index];
                }
            }
        }
    }
}

void CUUnifiedBlob::convolutionalBackPropFromError(CUUnifiedBlob &error, CUUnifiedBlob &filters,
                                                   CUUnifiedBlob &delta_filters,
                                                   CUUnifiedBlob &originalInput, CUUnifiedBlob &newErrorGradient,
                                                   int iHeight, int iWidth, int fHeight,
                                                   int fWidth, int depth, int numFilters) {
    int outputHeight = iHeight - fHeight + 1;
    int outputWidth = iWidth - fWidth + 1;

    for (int ch = 0; ch < numFilters; ch++) {
        for (int r = 0; r < outputHeight; r++) {
            for (int c = 0; c < outputWidth; c++) {
                int dh_index = ch * outputHeight * outputWidth + r * outputWidth + c;
                double dh = error.data[dh_index];

                for (int inputCh = 0; inputCh < depth; inputCh++) {
                    for (int fr = 0; fr < fHeight; fr++) {
                        for (int fc = 0; fc < fWidth; fc++) {
                            int inputIndex = (inputCh * iHeight * iWidth) + ((fr + r) * iWidth) + (fc + c);
                            int filterIndex = (ch * depth * fHeight * fWidth) + (inputCh * fHeight * fWidth) + (fr * fWidth) + fc;
                            // get the new error gradient (if it matters at all)
                            newErrorGradient.data[inputIndex] += filters.data[filterIndex] * dh;
                            // update the filters delta, but don't apply it to the actual filter until later
                            delta_filters.data[filterIndex] += originalInput.data[inputIndex] * dh;
                        }
                    }
                }
            }
        }
    }
}

void CUUnifiedBlob::getSquaredLength(CUUnifiedBlob &v, CUUnifiedBlob &lengths, int numClasses, int dim) {
    for (int v_index = 0; v_index < numClasses; v_index++) {
        double sq_length = 0;
        for (int i = 0; i < dim; i++) {
            sq_length += v.data[v_index*dim+i]*v.data[v_index*dim+i];
        }
        lengths.data[v_index] = sq_length;
    }
}

void CUUnifiedBlob::getVectorLoss(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, CUUnifiedBlob &losses, int numClasses,
                                  int dim) {
    const double m_plus = 0.9;
    const double m_minus = 0.1;
    const double lambda = 0.5;

    arma::vec currentV(dim);
    for (int v_index = 0; v_index < numClasses; v_index++) {
        for (int d = 0; d < dim; d++) {
            currentV[d] = v.data[v_index*dim + d];
        }
        double vec_length = Utils::length(currentV);

        double activationFactor = 2 * vec_length / pow((vec_length * vec_length) + 1, 2);

        double errorGradient;
        if (vec_length < m_plus) {
            if (vec_length <= m_minus) {
                errorGradient = -2 * truthMap.data[v_index] * (m_plus - vec_length);
            } else {
                errorGradient = 2 * ((lambda * (truthMap.data[v_index] - 1) * (m_minus - vec_length)) +
                                     truthMap.data[v_index] * (vec_length - m_plus));
            }
        } else {
            errorGradient = 2 * lambda * (truthMap.data[v_index] - 1) * (m_minus - vec_length);
        }

        double rawMarginLoss;
        if (truthMap.data[v_index]) {
            rawMarginLoss = pow(std::max(0.0, m_plus - vec_length), 2);
        } else {
            rawMarginLoss = lambda * pow(std::max(0.0, vec_length - m_minus), 2);
        }

        losses.data[v_index] = activationFactor * errorGradient * rawMarginLoss / vec_length;
    }
}

void CUUnifiedBlob::CUDA_matrixVectorMultiplication(CUUnifiedBlob &matrix,
                                                    CUUnifiedBlob &inputVector,
                                                    CUUnifiedBlob &outputVector,
                                                    int inputDim,
                                                    int outputDim,
                                                    int numClasses,
                                                    int tensorSize) {
    dim3 multElementsBlocks(tensorSize, numClasses);
    cu_matrixVectorMultiplication_kernel <<< multElementsBlocks, outputDim >>> (matrix.data,
            inputVector.data,
            outputVector.data,
            inputDim,
            outputDim);
}

void CUUnifiedBlob::CUDA_vectorVectorSoftmax(CUUnifiedBlob &b,
                                             CUUnifiedBlob &c,
                                             int numClasses,
                                             int tensorSize) {
    int numThreads = std::min(1024, tensorSize);
    cu_vectorVectorSoftmax_kernel << < numClasses, numThreads, numThreads * sizeof(double) >> >
                                                               (b.data, c.data, numClasses, tensorSize);
}

void CUUnifiedBlob::CUDA_weightReduceVectors(CUUnifiedBlob &u_hat,
                                             CUUnifiedBlob &c,
                                             CUUnifiedBlob &v,
                                             int numClasses,
                                             int tensorSize,
                                             int dim) {
    dim3 blockDimensions(numClasses, dim);
    int numThreads = std::min(1024, tensorSize);
    cu_weightReduceVector_kernel << < blockDimensions, numThreads, numThreads * sizeof(double) >> >
                                                                   (u_hat.data, c.data, v.data, numClasses, tensorSize, dim);
}

void CUUnifiedBlob::CUDA_vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim) {
    cu_vectorSquash_kernel << < numVecs, vecDim, vecDim * sizeof(double) >> > (v.data, numVecs, vecDim);
}

void
CUUnifiedBlob::CUDA_vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses,
                                              int tensorSize, int dim) {
    dim3 blockDims(tensorSize, numClasses);
    cu_vectorVectorScalarProduct_kernel <<< blockDims, dim, dim * sizeof(double) >>>
                                                             (u_hat.data, v.data, b.data, numClasses, tensorSize, dim);
}

void CUUnifiedBlob::CUDA_vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim) {
    cu_vectorLossFunction_kernel <<< numClasses, dim, dim * sizeof(double) >>> (v.data, truthMap.data);
}

void CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &c, CUUnifiedBlob &w,
                                                    CUUnifiedBlob &v_error, int numClasses, int tensorSize,
                                                    int innerDim, int outerDim) {
    dim3 blockDims(tensorSize, numClasses);
    cu_weightedTransMatrixVecMult_kernel <<< blockDims, innerDim >>>
                                                                   (delta_u.data, c.data, w.data, v_error.data, innerDim, outerDim);
}

void CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(CUUnifiedBlob &w, CUUnifiedBlob &v_error, CUUnifiedBlob &old_u,
                                                         int numClasses, int tensorSize, int innerDim, int outerDim) {
    dim3 blockDims(tensorSize, numClasses);
    dim3 threadDims(outerDim, innerDim);
    cu_vectorVectorMatrixProductAndSum_kernel <<< blockDims, threadDims >>>
                                                              (w.data, v_error.data, old_u.data, numClasses, tensorSize, innerDim, outerDim);
}

void CUUnifiedBlob::CUDA_multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim) {
    dim3 blockDims(tensorSize);
    cu_multiVectorReduction_kernel <<< blockDims, dim >>> (u.data, numClasses, dim);
}

void CUUnifiedBlob::CUDA_elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_error, CUUnifiedBlob &w_velocity, int size) {
    cu_elementWiseErrorUpdate_kernel <<< size, 1 >>> (w.data, w_error.data, w_velocity.data);
}

void CUUnifiedBlob::CUDA_vectorSquashDerivative(CUUnifiedBlob &v, int numVecs, int vecDim, int numClasses) {
    cu_vectorSquashDerivative_kernel <<< numVecs, vecDim, vecDim * sizeof(double) >>> (v.data, numClasses);
}

void CUUnifiedBlob::CUDA_convolutionalDotProduct(CUUnifiedBlob &input, CUUnifiedBlob &filter, CUUnifiedBlob &output,
                                                 int iHeight, int iWidth, int fHeight, int fWidth, int depth,
                                                 int numFilters) {
    int outputHeight = iHeight - fHeight + 1;
    int outputWidth = iWidth - fWidth + 1;

    dim3 blockDims(numFilters, outputHeight, outputWidth);
    dim3 threadDims(depth, fHeight, fWidth);
    int numThreads = depth * fHeight * fWidth;

    cu_convolutionalDotProduct_kernel << < blockDims, threadDims, numThreads * sizeof(double) >> >
                                                                  (input.data, filter.data, output.data, iHeight, iWidth);
}

void CUUnifiedBlob::CUDA_tensorFlatteningAndActivatedRemapping(CUUnifiedBlob &flattenedTensor, CUUnifiedBlob &tensor,
                                                               int height, int width, int depth, int numClasses,
                                                               int dim) {
    dim3 blockDims(depth, height, width);
    dim3 threadDims(dim);
    cu_tensorFlatteningAndActivatedRemapping_kernel <<< blockDims, threadDims, dim * sizeof(double) >> >
                                                                                (flattenedTensor.data, tensor.data, numClasses);
}

void CUUnifiedBlob::CUDA_reconstructingTensorFromError(CUUnifiedBlob &tensor, CUUnifiedBlob &flattenedTensor,
                                                       int height, int width, int depth, int numClasses, int dim) {
    dim3 blockDims(depth, height, width);
    dim3 threadDims(dim);
    cu_reconstructingTensorFromError_kernel <<< blockDims, threadDims >>>
                                                            (tensor.data, flattenedTensor.data, numClasses);
}

void CUUnifiedBlob::CUDA_convolutionalBackPropFromError(CUUnifiedBlob &error,
                                                        CUUnifiedBlob &filters, CUUnifiedBlob &delta_filters,
                                                        CUUnifiedBlob &originalInput, CUUnifiedBlob &newErrorGradient,
                                                        int iHeight, int iWidth,
                                                        int fHeight, int fWidth,
                                                        int depth, int numFilters) {
    int outputHeight = iHeight - fHeight + 1;
    int outputWidth = iWidth - fWidth + 1;

    dim3 blockDims(numFilters, outputHeight, outputWidth);
    dim3 threadDims(depth, fHeight, fWidth);

    cu_convolutionalBackPropFromError_kernel <<<blockDims,threadDims >>>
                                                             (error.data, filters.data, delta_filters.data, originalInput.data, newErrorGradient.data, iHeight, iWidth);
}

void CUUnifiedBlob::CUDA_getSquaredLength(CUUnifiedBlob &v, CUUnifiedBlob &lengths, int numClasses, int dim) {
    cu_getSquaredLength_kernel<<<numClasses, dim, dim*sizeof(double)>>>(v.data, lengths.data);
}

void CUUnifiedBlob::CUDA_getVectorLoss(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, CUUnifiedBlob &losses, int numClasses,
                                       int dim) {
    cu_getVectorLoss_kernel<<<numClasses, dim, dim*sizeof(double)>>>(v.data, truthMap.data, losses.data);
}


__device__
double atomicAdd(double *addr, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) addr;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__
double sharedMemoryReduce(double *shared_mem, double thread_val) {
    return 0.0;
}

__global__
void cu_clearOut_kernel(double *data) {
    data[blockIdx.x] = 0;
}

__global__
void cu_matrixVectorMultiplication_kernel(double *matrix, double *inputVector, double *outputVector,
                                          int inputDim, int outputDim) {
    int t = blockIdx.x;
    int k = blockIdx.y;
    int r = threadIdx.x;
    int numClasses = gridDim.y;

    int element_index = t*numClasses + k;
    int input_index = element_index*inputDim;
    int matrix_index = element_index*inputDim*outputDim;
    int output_index = element_index*outputDim;

    double cache = 0.0;
    for (int c = 0; c < inputDim; c++) {
        if (!isnan(matrix[r*inputDim + c + matrix_index] * inputVector[c + input_index])) {
            cache += matrix[r*inputDim + c + matrix_index] * inputVector[c + input_index];
        }
    }
    outputVector[r+output_index] = cache;
}

__global__
void cu_vectorVectorSoftmax_kernel(double *b, double *c, int numClasses, int tensorSize) {
    int t = threadIdx.x;
    int k = blockIdx.x;

    double my_exp_bs[3]; // make this dynamic and only as needed
    extern __shared__
    double shared_b_exps[];

    shared_b_exps[t] = 0.0;
    for (int i = 0; (i * 1024) < tensorSize; i++) {
        int tensorRowIndex = (i*1024) + t;
        int b_index = (tensorRowIndex * numClasses) + k;

        my_exp_bs[i] = exp(b[b_index]); // consider using hexp() for speed
        if (isnan(my_exp_bs[i])) {
            my_exp_bs[i] = 0;
        }
        shared_b_exps[t] += my_exp_bs[i];
    }
    __syncthreads();

    if (isinf(shared_b_exps[t])) {
        shared_b_exps[t] = 0;
    }
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (t % (2 * s) == 0 && (t+s) < blockDim.x && !isinf(shared_b_exps[t] + shared_b_exps[t+s])) {
            shared_b_exps[t] += shared_b_exps[t + s];
        }
        __syncthreads();
    }

    for (int i = 0; i * 1024 < tensorSize; i++) {
        int tensorRowIndex = i*1024 + t;
        c[tensorRowIndex * numClasses + k] = my_exp_bs[i] / shared_b_exps[0];
        if (isnan(c[tensorRowIndex * numClasses + k])) {
            c[tensorRowIndex * numClasses + k] = 0;
        }
    }
}

__global__
void cu_weightReduceVector_kernel(double *u_hat, double *c, double *v, int numClasses, int tensorSize, int dim) {
    int k = blockIdx.x;
    int specificDim = blockIdx.y;
    int t = threadIdx.x;

    int u_hat_index = t * numClasses * dim + k * dim;
    int c_index = t * numClasses + k;
    extern __shared__
    double shared_v_vec[];

    shared_v_vec[t] = 0;

    // if tensorsize > 1024, add them to the shared mem as well
    for (int i = 0; i * 1024 < tensorSize; i++) {
        int u_hat_offset = i * 1024 * numClasses * dim;
        int c_offset = i * 1024 * numClasses;
        shared_v_vec[t] += u_hat[u_hat_index + specificDim + u_hat_offset] * c[c_index + c_offset];
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (t % (2 * s) == 0 && (t+s) < blockDim.x && !isinf(shared_v_vec[t] + shared_v_vec[t+s])) {
            shared_v_vec[t] += shared_v_vec[t + s];
        }
        __syncthreads();
    }

    if (t == 0) {
        v[k * dim + specificDim] = shared_v_vec[0];
    }
}

__global__
void cu_vectorSquash_kernel(double *v, int numVecs, int vecDim) {
    int v_index = blockIdx.y * gridDim.x + blockIdx.x;
    int v_val_index = threadIdx.x;

    extern __shared__
    double shared_v_values[];
    // reduce the square of the individual elements in shared mem
    if (v_index < numVecs) {
        shared_v_values[v_val_index] = v[v_index * vecDim + v_val_index]*v[v_index * vecDim + v_val_index];
    }
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (v_val_index % (2 * s) == 0) {
            shared_v_values[v_val_index] += shared_v_values[v_val_index + s];
        }
        __syncthreads();
    }

    // calc squashing func
    if (v_val_index == 0) {
        shared_v_values[0] += 1e-4;
        shared_v_values[1] = shared_v_values[0] / (1 + shared_v_values[0]);
        shared_v_values[0] = sqrt(shared_v_values[0]);
    }
    __syncthreads();

    if (v_index < numVecs) {
        v[v_index * vecDim + v_val_index] *= shared_v_values[1] / shared_v_values[0];
    }
}

__global__
void cu_vectorVectorScalarProduct_kernel(double *u_hat, double *v, double *b, int numClasses, int tensorSize, int dim) {
    int k = blockIdx.y;
    int d = threadIdx.x;
    int t = blockIdx.x;
    int u_hat_index = t * numClasses * dim + k * dim;
    int v_index = k * dim;
    int b_index = t * numClasses + k;

    extern __shared__
    double shared_scalar_products[];
    shared_scalar_products[d] = u_hat[u_hat_index + d] * v[v_index + d];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (d % (2 * s) == 0 && (d + s) < blockDim.x) {
            shared_scalar_products[d] += shared_scalar_products[d + s];
        }
        __syncthreads();
    }

    if (d == 0) {
        if (!isnan(shared_scalar_products[0])) {
            b[b_index] += shared_scalar_products[0];
        }
    }
}

__global__
void cu_vectorLossFunction_kernel(double *v, double *truthMap) {
    const double m_plus = 0.9;
    const double m_minus = 0.1;
    const double lambda = 0.5;

    int k = blockIdx.x;
    int d = threadIdx.x;
    int dim = blockDim.x;

    extern __shared__
    double shared_vector_lengths[];
    shared_vector_lengths[d] = v[k * dim + d] * v[k * dim + d];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (d % (2 * s) == 0) {
            shared_vector_lengths[d] += shared_vector_lengths[d + s];
        }
        __syncthreads();
    }
    if (d == 0) {
        shared_vector_lengths[0] = sqrt(shared_vector_lengths[0] + 1e-4); // length of vector

        double activationFactor = 2 * shared_vector_lengths[0] /
                                  ((shared_vector_lengths[0] * shared_vector_lengths[0] + 1) *
                                   (shared_vector_lengths[0] * shared_vector_lengths[0] + 1));

        double errorGradient;
        if (shared_vector_lengths[0] < m_plus) {
            if (shared_vector_lengths[0] <= m_minus) {
                errorGradient = -2 * truthMap[k] * (m_plus - shared_vector_lengths[0]);
            } else {
                errorGradient = 2 * ((lambda * (truthMap[k] - 1) * (m_minus - shared_vector_lengths[0])) +
                                     truthMap[k] * (shared_vector_lengths[0] - m_plus));
            }
        } else {
            errorGradient = 2 * lambda * (truthMap[k] - 1) * (m_minus - shared_vector_lengths[0]);
        }

        double rawMarginLoss;
        if (truthMap[k] == 1.0) {
            rawMarginLoss = max(0.0, m_plus - shared_vector_lengths[0]);
            rawMarginLoss *= rawMarginLoss;
        } else {
            rawMarginLoss = max(0.0, shared_vector_lengths[0] - m_minus);
            rawMarginLoss *= rawMarginLoss;
            rawMarginLoss *= lambda;
        }
        // this is the new resizing factor
        shared_vector_lengths[1] = activationFactor * errorGradient * rawMarginLoss / shared_vector_lengths[0];
    }
    __syncthreads();


    v[k * dim + d] *= shared_vector_lengths[1];
}

__global__
void cu_weightedTransMatrixVecMult_kernel(double *delta_u, double *c, double *w, double *v_error, int innerDim,
                                          int outerDim) {
    int t = blockIdx.x;
    int k = blockIdx.y;

    int c_index = t * gridDim.y + k;
    int w_index = c_index * innerDim * outerDim;
    int u_index = c_index * innerDim;
    int v_index = k * outerDim;

    int col = threadIdx.x;

    double u_value = 0.0;
    for (int row = 0; row < outerDim; row++) {
        u_value += w[(row * innerDim + col) + w_index] * v_error[row + v_index];
    }
    delta_u[col + u_index] = u_value * c[c_index];
}

__global__
void
cu_vectorVectorMatrixProductAndSum_kernel(double *w_error, double *v_error, double *old_u, int numClasses, int tensorSize,
                                          int innerDim, int outerDim) {
    int t = blockIdx.x;
    int k = blockIdx.y;

    int row = threadIdx.x;
    int col = threadIdx.y;

    int element_index = t * numClasses + k;
    int w_index = element_index * innerDim * outerDim;
    int u_index = element_index * innerDim;
    int v_index = k * outerDim;

    w_error[row * innerDim + col + w_index] += v_error[row + v_index] * old_u[col + u_index];
}

__global__
void cu_multiVectorReduction_kernel(double *u, int numClasses, int dim) {
    int t = blockIdx.x;
    int d = threadIdx.x;

    for (int i = 1; i < numClasses; i++) {
        int u_element_index = t * numClasses + i;
        u[t * numClasses * dim + d] += u[u_element_index * dim + d];
        u[u_element_index * dim + d] = 0;
    }
}

__global__
void cu_elementWiseErrorUpdate_kernel(double *w, double *w_error, double *w_velocity) {
    int tid = blockIdx.x;
    w_velocity[tid] = 0.9*w_velocity[tid] + 0.1*w_error[tid];
    w[tid] -= w_velocity[tid];
    w_error[tid] = 0;
}

__global__
void cu_vectorSquashDerivative_kernel(double *v, int numClasses) {
    int t = blockIdx.x;
    int dim = blockDim.x;
    int specificDim = threadIdx.x;
    int row_dim = t * numClasses * dim;

    extern __shared__
    double shared_v_values[];
    // reduce the square of the individual elements in shared mem
    shared_v_values[specificDim] =  v[t * row_dim + specificDim] * v[t * row_dim + specificDim];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (specificDim % (2 * s) == 0) {
            shared_v_values[specificDim] += shared_v_values[specificDim + s];
        }
        __syncthreads();
    }

    // calc squashing func
    if (specificDim == 0) {
        shared_v_values[1] = (shared_v_values[0] + 1) * (shared_v_values[0] + 1);
        shared_v_values[0] += 1e-4;
        shared_v_values[0] = sqrt(shared_v_values[0]); // normalization factor

        shared_v_values[1] = 2 * shared_v_values[0] / shared_v_values[1]; // derivative factor
    }
    __syncthreads();

    v[t * row_dim + specificDim] *= shared_v_values[1] / shared_v_values[0];
}

__global__
void cu_convolutionalDotProduct_kernel(double *input, double *filter, double *output, int iHeight, int iWidth) {
    int f = blockIdx.x;
    int r = blockIdx.y;
    int c = blockIdx.z;

    int ch = threadIdx.x;
    int fr = threadIdx.y;
    int fc = threadIdx.z;

    int outputHeight = iHeight - blockDim.y + 1;
    int outputWidth = iWidth - blockDim.z + 1;

    int tid = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;

    int filterIndex = f * blockDim.x * blockDim.y * blockDim.z + ch * blockDim.y * blockDim.z + fr * blockDim.z + fc;
    int inputIndex = ch * iHeight * iWidth + (r + fr) * iWidth + (c + fc);
    int outputIndex = f * outputHeight * outputWidth + r * outputWidth + c;

    extern __shared__
    double shared_matrix_dot[];

    shared_matrix_dot[tid] = input[inputIndex] * filter[filterIndex];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x * blockDim.y * blockDim.z; s *= 2) {
        if (tid % (2 * s) == 0) {
            shared_matrix_dot[tid] += shared_matrix_dot[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_matrix_dot[0] /= blockDim.x * blockDim.y * blockDim.z;
        output[outputIndex] = shared_matrix_dot[0];
//        output[outputIndex] = filterIndex;
    }
}

__global__
void cu_tensorFlatteningAndActivatedRemapping_kernel(double *flattenedTensor, double *tensor, int numClasses) {
    int height = gridDim.y;
    int width = gridDim.z;
    int depth = gridDim.x;
    int dim = blockDim.x;

    int r = blockIdx.y;
    int c = blockIdx.z;
    int d = blockIdx.x;

    int dimIndex = threadIdx.x;
    int vector_index = r * depth * width + c * depth + d;
    int tensor_index = (d * dim + dimIndex) * height * width + r * width + c;

    extern __shared__
    double shared_v_values[];

    shared_v_values[dimIndex] = tensor[tensor_index] * tensor[tensor_index];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (dimIndex % (2 * s) == 0) {
            shared_v_values[dimIndex] += shared_v_values[dimIndex + s];
        }
        __syncthreads();
    }

    if (dimIndex == 0) {
        shared_v_values[0] += 1e-4;
        shared_v_values[1] = shared_v_values[0] / (1 + shared_v_values[0]);
        shared_v_values[0] = sqrt(shared_v_values[0]);
    }
    __syncthreads();

    for (int k = 0; k < numClasses; k++) {
        int output_index = vector_index * (numClasses * dim) + (dimIndex + (k * dim));
        flattenedTensor[output_index] = tensor[tensor_index] * shared_v_values[1] / shared_v_values[0];
    }
}

__global__
void cu_reconstructingTensorFromError_kernel(double *tensor, double *flattenedTensor, int numClasses) {
    int depth = gridDim.x;
    int height = gridDim.y;
    int width = gridDim.z;
    int dim = blockDim.x;

    int r = blockIdx.y;
    int c = blockIdx.z;
    int d = blockIdx.x;

    int dimIndex = threadIdx.x;
    int vector_index = r * depth * width + c * width + d;
    int tensor_index = (d * dim + dimIndex) * height * width + r * width + c;
    int error_index = (vector_index) * (numClasses*dim) + dimIndex;

    tensor[tensor_index] = flattenedTensor[error_index];
}

__global__
void cu_convolutionalBackPropFromError_kernel(double *error, double *filters, double *delta_filters,
                                              double *originalInput, double *newErrorGradient, int iHeight,
                                              int iWidth) {
    int ch = blockIdx.x;
    int r = blockIdx.y;
    int c = blockIdx.z;

    int inputCh = threadIdx.x;
    int fr = threadIdx.y;
    int fc = threadIdx.z;

    int dh_index = ch * gridDim.y * gridDim.z + r * gridDim.z + c;
    double dh = error[dh_index];

    int inputIndex = (inputCh * iHeight * iWidth) + ((fr + r) * iWidth) + (fc + c);
    int filterIndex = (ch * blockDim.x * blockDim.y * blockDim.z) + (inputCh * blockDim.y * blockDim.z) + (fr * blockDim.z) + fc;

    //newErrorGradient[inputIndex] += filters[filterIndex] * dh;
    atomicAdd(&newErrorGradient[inputIndex], filters[filterIndex] * dh);
    //filter_error[filterIndex] += originalInput[inputIndex] * dh;
    atomicAdd(&delta_filters[filterIndex], originalInput[inputIndex] * dh);
}

__global__
void cu_getSquaredLength_kernel(double *v, double *lengths) {
    int v_index = blockIdx.x;
    int specificDim = threadIdx.x;
    int dim = blockDim.x;

    extern __shared__
    double shared_v_values[];

    shared_v_values[specificDim] = v[v_index*dim+specificDim] * v[v_index*dim+specificDim];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (specificDim % (2 * s) == 0) {
            shared_v_values[specificDim] += shared_v_values[specificDim + s];
        }
        __syncthreads();
    }

    // find out who's the biggest, and save his value in the first number
    if (specificDim == 0) {
        lengths[v_index] = shared_v_values[0];
    }
}

__global__
void cu_getVectorLoss_kernel(double *v, double *truthMap, double *losses) {
    const double m_plus = 0.9;
    const double m_minus = 0.1;
    const double lambda = 0.5;

    int k = blockIdx.x;
    int d = threadIdx.x;
    int dim = blockDim.x;

    extern __shared__
    double shared_vector_lengths[];
    shared_vector_lengths[d] = v[k * dim + d] * v[k * dim + d];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (d % (2 * s) == 0) {
            shared_vector_lengths[d] += shared_vector_lengths[d + s];
        }
        __syncthreads();
    }
    if (d == 0) {
        shared_vector_lengths[0] = sqrt(shared_vector_lengths[0] + 1e-4); // length of vector
    }
    __syncthreads();

    double rawMarginLoss;
    if (truthMap[k]) {
        rawMarginLoss = max(0.0, m_plus - shared_vector_lengths[0]);
        rawMarginLoss *= rawMarginLoss;
    } else {
        rawMarginLoss = max(0.0, shared_vector_lengths[0] - m_minus);
        rawMarginLoss *= rawMarginLoss * lambda;
    }

    double resizingFactor = rawMarginLoss;
    losses[k] = resizingFactor;
}