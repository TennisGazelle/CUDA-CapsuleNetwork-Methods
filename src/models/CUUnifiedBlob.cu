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
    CUDA_clear();
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
    error = cudaMallocManaged((void **) &flagHelper, sizeof(int));
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
    cu_clearOut_kernel <<< size, 1 >>> (data);
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::resize(int newSize) {
    deallocateMemory();
    size = newSize;
    allocateMemory();
}

void CUUnifiedBlob::copy(const CUUnifiedBlob &other) {
    cudaDeviceSynchronize();
    if (other.size != size) {
        resize(other.size);
    }
    for (int i = 0; i < size; i++) {
        data[i] = other.data[i];
    }
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::print(const std::string &msg, int width) const {
    cudaDeviceSynchronize();
    if (!msg.empty()) {
        std::cout << msg << ", size: " << size << std::endl;
    }
    int bufferSize = std::min(size, 10000);
    for (int i = 0; i < bufferSize; i++) {
        std::cout << data[i] << "\t";
        if (((i + 1) % width) == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl << std::endl;
//    for (int i = 0; i < size; i++) {
//        if (data[i] == 0.0) {
//            std::cout << "zero at location: " << i << "(" << i/width << "," << i % width << ")" << std::endl;
//        }
//    }
}

CUUnifiedBlob& CUUnifiedBlob::operator=(const CUUnifiedBlob &other) {
    if (this == &other) {
        return *this;
    }

    resize(other.size);
    cudaDeviceSynchronize();
    for (int i = 0; i < size; i++) {
        data[i] = other.data[i];
    }
    return *this;
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
        data[i] = Utils::getWeightRand(0);
    }
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::fillSequentially() {
    for (int i = 0; i < size; i++) {
        data[i] = i;
    }
}

int CUUnifiedBlob::hasNan() const {
    cudaDeviceSynchronize();
    for (int i = 0; i < size; i++) {
        if (isnan(data[i])) {
            return i;
        }
    }
    return -1;
}

bool CUUnifiedBlob::CUDA_hasNan() const {
    flagHelper[0] = 0;
    cu_hasNan_kernel<<<size, 1>>>(data, flagHelper);
    cudaDeviceSynchronize();
    bool toReturn = (flagHelper[0] != 0);
    flagHelper[0] = 0;
    return toReturn;
}

int CUUnifiedBlob::hasInf() const {
    cudaDeviceSynchronize();
    for (int i = 0; i < size; i++) {
        if (isinf(data[i])) {
            return i;
        }
    }
    return -1;
}

bool CUUnifiedBlob::isAllZeros() const {
    cudaDeviceSynchronize();
    for (int i = 0; i < size; i++) {
        if (data[i] != 0.0) {
            return false;
        }
    }
    return true;
}

double CUUnifiedBlob::getValueAt_1D(int location) const {
    assert(0 <= location && location < size);
    cudaDeviceSynchronize();
    return data[location];
}

double CUUnifiedBlob::getValueAt_2D(int x, int y, int xDim) const {
    int location = x*xDim + y;
    return getValueAt_1D(location);
}

void CUUnifiedBlob::setValueAt_1D(int location, double incomingValue) {
    assert(location >= 0 && location < size);
    data[location] = incomingValue;
}

void CUUnifiedBlob::CUDA_setValueAt(int location, double incomingValue) {
    assert(location >= 0 && location < size);
    cu_singleElementSetting_kernel<<<size, 1>>>(data, location, incomingValue);
    cudaDeviceSynchronize();
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

void CUUnifiedBlob::vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim, double m_plus,
                                       double m_minus, double lambda) {
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

void CUUnifiedBlob::elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_delta, CUUnifiedBlob &w_velocity, int size) {
    cudaDeviceSynchronize();
    for (int i = 0; i < size; i++) {
        w_velocity.data[i] = w_velocity.data[i]*0.9 + w_delta.data[i]*0.1;
        w.data[i] += w_velocity.data[i];
        w_delta.data[i] = 0;
    }
    CUDAUtils::checkForError("inside the elementWise Update");
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
    int outputHeight = iHeight - fHeight + 1;
    int outputWidth = iWidth - fWidth + 1;

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
                output.setValueAt_1D(outputIndex, max(0.0, sum));
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
                                                  int width, int vectorDepth, int numClasses, int dim) {
    // go through each of the vector (filterDepth first, then width, then height)
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            for (int d = 0; d < vectorDepth; d++) {
                // collect error from the thing
                int vector_index =
                        r * (vectorDepth * width) +
                        c * width +
                        d;

                for (int dimIndex = 0; dimIndex < dim; dimIndex++) {
                    int error_index = (vector_index) * (numClasses * dim) + dimIndex;
                    int realDepth = d*dim + dimIndex;
                    int tensor_index =
                            realDepth * height * width +
                            r * width +
                            c;

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

    int highestFilter = 0;
    int highestDH = 0;
    int highestInput = 0;

    for (int ch = 0; ch < numFilters; ch++) {
        for (int r = 0; r < outputHeight; r++) {
            for (int c = 0; c < outputWidth; c++) {
                int dh_index = ch * outputHeight * outputWidth + r * outputWidth + c;
                highestDH = max(dh_index, highestDH);
                double dh = error.data[dh_index];

                for (int inputCh = 0; inputCh < depth; inputCh++) {
                    for (int fr = 0; fr < fHeight; fr++) {
                        for (int fc = 0; fc < fWidth; fc++) {
                            int inputIndex = (inputCh * iHeight * iWidth) + ((fr + r) * iWidth) + (fc + c);
                            int filterIndex = (ch * depth * fHeight * fWidth) + (inputCh * fHeight * fWidth) + (fr * fWidth) + fc;
                            highestInput = max(inputIndex, highestInput);
                            highestFilter = max(filterIndex, highestFilter);
                            // get the new error gradient (if it matters at all)
                            //newErrorGradient.data[inputIndex] += filters.data[filterIndex] * dh;
                            // update the filters delta, but don't apply it to the actual filter until later
                            delta_filters.data[filterIndex] += originalInput.data[inputIndex] * dh;
                        }
                    }
                }
            }
        }
    }

//    std::cout << "highest filter index: " << highestFilter << ", max filter is: " << delta_filters.size << std::endl;
//    std::cout << "highest input index : " << highestInput << ", max input  is:" << originalInput.size << std::endl;
//    std::cout << "highest dh index    : " << highestDH << " max dh     is: " << error.size << std::endl;
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

void CUUnifiedBlob::getVectorLoss(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, CUUnifiedBlob &losses, int numClasses, int dim,
                                  double m_plus, double m_minus, double lambda) {

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
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_vectorVectorSoftmax(CUUnifiedBlob &b,
                                             CUUnifiedBlob &c,
                                             int numClasses,
                                             int tensorSize) {
    int numThreads = std::min(1024, tensorSize);
    cu_vectorVectorSoftmax_kernel <<< numClasses, numThreads, numThreads * sizeof(double) >>>
                                                               (b.data, c.data, numClasses, tensorSize);
    cudaDeviceSynchronize();
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_vectorVectorSoftmax()");
}

void CUUnifiedBlob::CUDA_weightReduceVectors(CUUnifiedBlob &u_hat,
                                             CUUnifiedBlob &c,
                                             CUUnifiedBlob &v,
                                             int numClasses,
                                             int tensorSize,
                                             int dim) {
    dim3 blockDimensions(numClasses, dim);
    cu_weightReduceVector_kernel <<< blockDimensions, 1 >>> (u_hat.data, c.data, v.data, numClasses, tensorSize, dim);
    cudaDeviceSynchronize();
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_weightReduceVector()");
}

void CUUnifiedBlob::CUDA_vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim) {
    cu_vectorSquash_kernel <<< numVecs, vecDim, vecDim * sizeof(double) >>> (v.data, numVecs, vecDim);
    cudaDeviceSynchronize();
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_vectorSquash()");
}

void
CUUnifiedBlob::CUDA_vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses,
                                              int tensorSize, int dim) {
    dim3 blockDims(tensorSize, numClasses);
    cu_vectorVectorScalarProduct_kernel <<< blockDims, dim, dim * sizeof(double) >>>
                                                             (u_hat.data, v.data, b.data, numClasses, tensorSize, dim);
    cudaDeviceSynchronize();
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_vectorVectorScalarProduct()");
}

void CUUnifiedBlob::CUDA_vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim,
                                           double m_plus, double m_minus, double lambda) {
    cu_vectorLossFunction_kernel <<< numClasses, dim, dim * sizeof(double) >>> (v.data, truthMap.data, m_plus, m_minus, lambda);
    cudaDeviceSynchronize();
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_vectorLossFunction()");
}

void CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &w,
                                                    CUUnifiedBlob &delta_u_hat_error, int numClasses, int tensorSize,
                                                    int innerDim, int outerDim) {
    dim3 blockDims(tensorSize, numClasses);
    cu_weightedTransMatrixVecMult_kernel <<< blockDims, innerDim >>>
                                                                   (delta_u.data, w.data, delta_u_hat_error.data, innerDim, outerDim);
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_weightedTransMatrixVecMult");
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_scaledDecompositionOfError(CUUnifiedBlob &delta_v, CUUnifiedBlob &c, CUUnifiedBlob &delta_u_hat, int numClasses, int tensorSize, int dim) {
	dim3 blockDims(tensorSize, numClasses);
	dim3 threadDims(dim);

	cu_scaledDecompositionOfError<<<blockDims, threadDims>>>(delta_v.data, c.data, delta_u_hat.data, numClasses);
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_scaledDecompositionOfError()");
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(CUUnifiedBlob &w_delta, CUUnifiedBlob &delta_v, CUUnifiedBlob &old_u,
                                                         int numClasses, int tensorSize, int innerDim, int outerDim) {
    dim3 blockDims(tensorSize, numClasses);
    dim3 threadDims(outerDim, innerDim);
    cu_vectorVectorMatrixProductAndSum_kernel <<< blockDims, threadDims >>>
                                                              (w_delta.data, delta_v.data, old_u.data, numClasses, tensorSize, innerDim, outerDim);
    CUDAUtils::checkForError("CUUnifedBlob::CUDA_vectorVectorMatrixProductAndSum");
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim) {
    dim3 blockDims(tensorSize, dim);
    cu_multiVectorReduction_kernel <<< blockDims, numClasses, numClasses*sizeof(double) >>> (u.data, numClasses, dim);
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_elementWiseErrorUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_error, CUUnifiedBlob &w_velocity, int size) {
    cu_elementWiseErrorUpdate_kernel<<<size, 1>>>(w.data, w_error.data, w_velocity.data);
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_elementWiseErrorUpdate()");
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_vectorSquashDerivative(CUUnifiedBlob &v, int numVecs, int vecDim, int numClasses) {
    cu_vectorSquashDerivative_kernel <<< numVecs, vecDim, vecDim * sizeof(double) >>> (v.data, numClasses);
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_convolutionalDotProduct(CUUnifiedBlob &input, CUUnifiedBlob &filter, CUUnifiedBlob &output,
                                                 int iHeight, int iWidth, int fHeight, int fWidth, int depth,
                                                 int numFilters) {
    int outputHeight = iHeight - fHeight;
    int outputWidth = iWidth - fWidth;

    dim3 blockDims(numFilters, outputHeight, outputWidth);
    dim3 threadDims(depth, fHeight, fWidth);
    int numThreads = depth * fHeight * fWidth;

    cu_convolutionalDotProduct_kernel <<< blockDims, threadDims, numThreads * sizeof(double) >>>
                                                                  (input.data, filter.data, output.data, iHeight, iWidth);
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_convolutionalDotProduct");
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_tensorFlatteningAndActivatedRemapping(CUUnifiedBlob &flattenedTensor, CUUnifiedBlob &tensor,
                                                               int height, int width, int sDepth, int numClasses,
                                                               int dim) {
    dim3 blockDims(sDepth, height, width);
    cu_tensorFlatteningAndActivatedRemapping_kernel <<< blockDims, dim, dim * sizeof(double) >>>
                                                                                (flattenedTensor.data, tensor.data, numClasses);
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_reconstructingTensorFromError(CUUnifiedBlob &tensor, CUUnifiedBlob &flattenedTensor,
                                                       int height, int width, int vectorDepth, int numClasses, int dim) {
    dim3 blockDims(vectorDepth, height, width);
    dim3 threadDims(dim);
    cu_reconstructingTensorFromError_kernel <<< blockDims, threadDims >>>
                                                            (tensor.data, flattenedTensor.data, numClasses);
    CUDAUtils::checkForError("CUUnifiedBlob::CUDA_reconstructingTensorFromError()");
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_convolutionalBackPropFromError(CUUnifiedBlob &error,
                                                        CUUnifiedBlob &filters, CUUnifiedBlob &delta_filters,
                                                        CUUnifiedBlob &originalInput, CUUnifiedBlob &newErrorGradient,
                                                        int iHeight, int iWidth,
                                                        int fHeight, int fWidth,
                                                        int depth, int numFilters) {
    int outputHeight = iHeight - fHeight;
    int outputWidth = iWidth - fWidth;

    dim3 blockDims(numFilters, outputHeight, outputWidth);
    dim3 threadDims(depth, fHeight, fWidth);

    cu_convolutionalBackPropFromError_kernel <<<blockDims,threadDims >>>
                                                             (error.data, filters.data, delta_filters.data, originalInput.data, newErrorGradient.data, iHeight, iWidth);
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_getLength(CUUnifiedBlob &v, CUUnifiedBlob &lengths, int numClasses, int dim) {
    cu_getLength_kernel <<<numClasses, dim, dim*sizeof(double)>>>(v.data, lengths.data);
    cudaDeviceSynchronize();
}

void CUUnifiedBlob::CUDA_getVectorLoss(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, CUUnifiedBlob &losses, int numClasses,
                                       int dim, double m_plus, double m_minus, double lambda) {
    cu_getVectorLoss_kernel<<<numClasses, dim, dim*sizeof(double)>>>(v.data, truthMap.data, losses.data, m_plus, m_minus, lambda);
    cudaDeviceSynchronize();
}

//#ifdef ATOMIC_ADD_DEFINITION_REQUIRED
//#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//__device__
//double atomicAdd(double *addr, double val) {
//    unsigned long long int *address_as_ull = (unsigned long long int *) addr;
//    unsigned long long int old = *address_as_ull, assumed;
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val + __longlong_as_double(assumed)));
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}
//#endif

__device__
double sharedMemoryReduce(double *shared_mem, double thread_val, int kernelIndex, int sharedMemSize) {
    // load shared memory
    shared_mem[kernelIndex] = thread_val;
    __syncthreads();

    thread_val = isinf(thread_val) ? 0.0 : thread_val; // clear out if isinf

    // reduce
//    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
//        int index = 2*s*kernelIndex;
//
//        if ((index+s) < blockDim.x && !isinf(shared_mem[index] + shared_mem[index+s])) {
//            shared_mem[index] += shared_mem[index + s];
//        }
//        __syncthreads();
//    }

    // sequential reduction for good measure
    if (kernelIndex == 0) {
        for (unsigned int i = 1; i < sharedMemSize; i++) {
            if (!isinf(shared_mem[0] + shared_mem[i])) {
                shared_mem[0] += shared_mem[i];
            }
        }
    }
    __syncthreads();

    // return index [0]
    return shared_mem[0];
}

__global__
void cu_clearOut_kernel(double *data) {
    data[blockIdx.x] = 0;
}

__global__
void cu_hasNan_kernel(double *data, int *flag) {
    if (isnan(data[blockIdx.x])) {
        flag[0] = 1;
    }
}

__global__
void cu_singleElementSetting_kernel(double *data, int location, double incomingValue) {
    if (blockIdx.x == location) {
        data[blockIdx.x] = incomingValue;
    }
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
        cache += matrix[matrix_index + (r*inputDim + c)] * inputVector[input_index + c];
    }
    outputVector[output_index + r] = cache;
}

__global__
void cu_vectorVectorSoftmax_kernel(double *b, double *c, int numClasses, int tensorSize) {
    int t_max = blockDim.x;
    int t = threadIdx.x;
    int k = blockIdx.x;

    double my_exp_bs[16]; // make this dynamic and only as needed
    extern __shared__
    double shared_b_exps[];

    double initial_value = 0.0;
    int sharedIndex = 0;
    for (int i = t; i < tensorSize; i += t_max) {
        int b_index = (i * numClasses) + k;

        my_exp_bs[sharedIndex] = exp(b[b_index]); // consider using hexp() for speed
        if (isnan(my_exp_bs[sharedIndex]) || isinf(my_exp_bs[sharedIndex])) {
            my_exp_bs[sharedIndex] = 0;
        }
        initial_value += my_exp_bs[sharedIndex];
        sharedIndex++;
    }
    if (isinf(initial_value)) {
        printf("there is a softmax inf!!!");
    }

    double reduction = sharedMemoryReduce(shared_b_exps, initial_value, t, t_max);

    sharedIndex = 0;
    for (int i = t; i < tensorSize; i += t_max) {
        int c_index = (i * numClasses) + k;

        c[c_index] = my_exp_bs[sharedIndex] / reduction;
        if (isnan(c[c_index])) {
            c[c_index] = 0;
        }
        sharedIndex++;
    }
}

__global__
void cu_weightReduceVector_kernel(double *u_hat, double *c, double *v, int numClasses, int tensorSize, int dim) {
    int k = blockIdx.x;
    int specificDim = blockIdx.y;

    double cache = 0.0;
    for (int t = 0; t < tensorSize; t++) {
        int c_index = t * numClasses + k;
        int u_hat_index = c_index * dim;
        cache += u_hat[u_hat_index + specificDim] * c[c_index];
    }
    int v_index = k * dim + specificDim;
    v[v_index] = cache;
}

__global__
void cu_vectorSquash_kernel(double *v, int numVecs, int vecDim) {
    int vectorIndex = blockIdx.x;
    int specificDim = threadIdx.x;

    extern __shared__
    double shared_v_values[];
    // reduce the square of the individual elements in shared mem
    double initialValue = v[vectorIndex * vecDim + specificDim]*v[vectorIndex * vecDim + specificDim];

    sharedMemoryReduce(shared_v_values, initialValue, specificDim, vecDim);

    // calc squashing func
    if (specificDim == 0) {
        shared_v_values[0] += EPSILON;
        shared_v_values[1] = shared_v_values[0] / (1 + shared_v_values[0]);
        shared_v_values[0] = sqrt(shared_v_values[0]+EPSILON);
    }
    __syncthreads();

    if (vectorIndex < numVecs) {
        v[vectorIndex * vecDim + specificDim] *= shared_v_values[1] / shared_v_values[0];
    }
}

__global__
void cu_vectorVectorScalarProduct_kernel(double *u_hat, double *v, double *b, int numClasses, int tensorSize, int dim) {
    int t = blockIdx.x;
    int k = blockIdx.y;
    int d = threadIdx.x;

    int b_index = t * numClasses + k;
    int u_hat_index = b_index * dim;
    int v_index = k * dim;

    extern __shared__
    double shared_scalar_products[];

    double product = sharedMemoryReduce(shared_scalar_products, u_hat[u_hat_index + d] * v[v_index + d], d, dim);

    if (d == 0) {
        if (!isnan(shared_scalar_products[0]) && !isinf(shared_scalar_products[0])) {
            b[b_index] += product;
        }
    }
}

__global__
void cu_vectorLossFunction_kernel(double *v, double *truthMap, double m_plus, double m_minus, double lambda) {
    int k = blockIdx.x;
    int d = threadIdx.x;
    int dim = blockDim.x;

    const double t_k = truthMap[k] > 0 ? 1.0 : 0.0;

    extern __shared__
    double shared_vector_lengths[];

    sharedMemoryReduce(shared_vector_lengths, v[k * dim + d] * v[k * dim + d], d, dim);
    
    if (d == 0) {
        const double l = sqrt(shared_vector_lengths[0] + EPSILON + EPSILON); // length of vector

        double activationFactor = pow(l*l + 1, 2);
        activationFactor = 2*l/activationFactor;

        double errorGradient = 0.0;
        if (l < m_plus) {
            if (l <= m_minus) {
                errorGradient = -2 * t_k * (m_plus - l);
            } else {
                errorGradient = 2 * ((lambda * (t_k - 1) * (m_minus - l)) + t_k * (l - m_plus));
            }
        } else {
            errorGradient = 2 * lambda * (t_k - 1) * (m_minus - l);
        }

        double rawMarginLoss = 0.0;
        if (t_k >= 1.0) {
            rawMarginLoss = max(0.0, m_plus - l);
            rawMarginLoss *= rawMarginLoss;
        } else {
            rawMarginLoss = pow(max(0.0, l - m_minus), 2);
            rawMarginLoss *= lambda;
        }
        // this is the new resizing factor
        shared_vector_lengths[1] = 0.1 * activationFactor * errorGradient * rawMarginLoss / sqrt(shared_vector_lengths[0]);
//        printf("raw sum of squares: %f\n", shared_vector_lengths[0]);
//        printf("for class %d, length: %f, t_k: %i, activation: %f, gradient: %f, loss: %f\n", k, l, t_k, activationFactor, errorGradient, rawMarginLoss);
    }

    __syncthreads();
    v[k * dim + d] *= shared_vector_lengths[1];
}

__global__
void cu_weightedTransMatrixVecMult_kernel(double *delta_u, double *w, double *delta_u_hat, int innerDim,
                                          int outerDim) {
    int t = blockIdx.x;
    int k = blockIdx.y;
    int c = threadIdx.x;
    int numClasses = gridDim.y;

    int element_index = t * numClasses + k;
    int u_index = element_index * innerDim;
    int w_index = element_index * innerDim * outerDim;
    int u_hat_index = element_index * outerDim;

    double u_value = 0.0;
    for (int r = 0; r < outerDim; r++) {
        u_value += w[w_index + (r * innerDim + c)] * delta_u_hat[u_hat_index + r];
    }
    delta_u[u_index + c] = u_value;
}

__global__
void cu_scaledDecompositionOfError(double *delta_v, double *c, double *delta_u_hat, int numClasses) {
	int t = blockIdx.x;
	int k = blockIdx.y;
	int d = threadIdx.x;

	int dim = blockDim.x;

	int c_index = t * numClasses + k;
	int delta_u_hat_index = c_index * dim;
    int delta_v_index = k * dim;

	delta_u_hat[delta_u_hat_index + d] = c[c_index] * delta_v[delta_v_index + d];
}

__global__
void cu_vectorVectorMatrixProductAndSum_kernel(double *w_error, double *delta_v, double *old_u, int numClasses, int tensorSize,
                                          int innerDim, int outerDim) {
    int t = blockIdx.x;
    int k = blockIdx.y;

    int row = threadIdx.x;
    int col = threadIdx.y;

    int element_index = t * numClasses + k;
    int w_index = element_index * innerDim * outerDim;
    int u_index = element_index * innerDim;
    int v_index = k * outerDim;

    w_error[w_index + (row * innerDim + col)] += delta_v[v_index + row] * old_u[u_index + col];
}

__global__
void cu_multiVectorReduction_kernel(double *u, int numClasses, int dim) {
    int t = blockIdx.x;
    int d = blockIdx.y;
    int k = threadIdx.x;

    int u_index = (t*numClasses + k)*dim + d;

    extern __shared__
    double shared_v_values[];

    if (t != k) {
        u[u_index] = 0;
    }

    sharedMemoryReduce(shared_v_values, u[u_index], k, numClasses); // TODO: investigate why I have to halve this value
    u[u_index] = 0;

//    if (k == 0) {
//        u[u_index] = shared_v_values[0];
//    }
    /**
     * This is persuant to a weird bug found in the sequential version.
     * The origin of this bug may be found in:
     * `src/CapsuleNetwork/CapsuleNetwork.cpp` in the back propagation
     * function.  This is not supposed to happen.
     */
    if (k == 0) {
    	u[u_index] = shared_v_values[0] * gridDim.x;
    }
}

__global__
void cu_elementWiseErrorUpdate_kernel(double *w, double *w_error, double *w_velocity) {
    const double velocity_coefficient = 0.9;
    int tid = blockIdx.x;
    if (isnan(w_error[tid])) {
        printf("there's a nan in the w somewhere, this is our worst nightmare\n");
    }
    w_velocity[tid] = velocity_coefficient*w_velocity[tid] + (1.0-velocity_coefficient)*w_error[tid];
    w[tid] -= w_velocity[tid];
    w_error[tid] = 0;
}

__global__
void cu_vectorSquashDerivative_kernel(double *v, int numClasses) {
    int t = blockIdx.x;
    int dim = blockDim.x;
    int specificDim = threadIdx.x;
    int v_index = t * numClasses * dim + specificDim;

    extern __shared__
    double shared_v_values[];
    // reduce the square of the individual elements in shared mem
    sharedMemoryReduce(shared_v_values, v[v_index] * v[v_index], specificDim, dim);

    // calc squashing func
    if (specificDim == 0) {
        if (shared_v_values[0] == 0.0) {
            shared_v_values[1] = 0.0;
        } else {
            shared_v_values[0] = sqrt(shared_v_values[0] + EPSILON + EPSILON); // normalization factor
            shared_v_values[1] = shared_v_values[0] * shared_v_values[0] + 1;
            shared_v_values[1] *= shared_v_values[1];
            shared_v_values[1] = 2 * shared_v_values[0] / shared_v_values[1]; // derivative factor
            shared_v_values[1] = shared_v_values[1] / shared_v_values[0];
        }
    }
    __syncthreads();

//    if (isnan(v[v_index] * shared_v_values[1] / shared_v_values[0])) {
//        if (specificDim == 0) {
//            printf("NAN is created for row %d in dim %d, which was originally %f\n", t, specificDim, v[v_index]);
//            printf("shared values[0]: %f, shared values [1]: %f\n", shared_v_values[0], shared_v_values[1]);
//        }
//    }
    v[v_index] *= shared_v_values[1];
}

__global__
void cu_convolutionalDotProduct_kernel(double *input, double *filter, double *output, int iHeight, int iWidth) {
    int depth = blockDim.x;
    int filterHeight = blockDim.y;
    int filterWidth = blockDim.z;

    int f = blockIdx.x;
    int r = blockIdx.y;
    int c = blockIdx.z;

    int ch = threadIdx.x;
    int fr = threadIdx.y;
    int fc = threadIdx.z;

    int outputHeight = gridDim.y;
    int outputWidth = gridDim.z;

    int tid = threadIdx.x * blockDim.y * blockDim.z
              + threadIdx.y * blockDim.z
              + threadIdx.z;

    int filterIndex = f * depth * filterHeight * filterWidth
                      + ch * filterHeight * filterWidth
                      + fr * filterWidth
                      + fc;
    int inputIndex = ch * iHeight * iWidth
                     + ((r + fr) * iWidth)
                     + (c + fc);
    int outputIndex = f * outputHeight * outputWidth
                      + r * outputWidth
                      + c;

    extern __shared__
    double shared_matrix_dot[];

    shared_matrix_dot[tid] = input[inputIndex] * filter[filterIndex];
    __syncthreads();

    if (tid == 0) {
        for (int i = 1; i < depth * filterHeight * filterWidth; i++) {
            shared_matrix_dot[0] += shared_matrix_dot[i];
        }
        shared_matrix_dot[0] /= depth * filterHeight * filterWidth;
        output[outputIndex] = shared_matrix_dot[0];
//        output[outputIndex] = max(0.0, shared_matrix_dot[0]);
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

    int tensor_index = (d * dim + dimIndex) * height * width + r * width + c;

    extern __shared__
    double shared_v_values[];

    sharedMemoryReduce(shared_v_values, tensor[tensor_index] * tensor[tensor_index], dimIndex, dim);

    if (dimIndex == 0) {
        shared_v_values[0] += EPSILON;
        shared_v_values[1] = shared_v_values[0] / (1 + shared_v_values[0]);
        shared_v_values[0] = sqrt(shared_v_values[0]+EPSILON);
    }
    __syncthreads();

    int vector_tensor_index = r * depth * width + c * depth + d;

    for (int k = 0; k < numClasses; k++) {
        int output_index = vector_tensor_index * (numClasses * dim) + (dimIndex + (k * dim));
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

    int vector_number = r*width*depth + c*depth + d;
    int error_index = vector_number * (numClasses * dim) + (dimIndex);

    int real_depth = d*dim + dimIndex;
    int tensor_index = real_depth * height * width + r * width + c;

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

    int depth = blockDim.x;
    int filterHeight = blockDim.y;
    int filterWidth = blockDim.z;

    int dh_index = ch * gridDim.y * gridDim.z + r * gridDim.z + c;
    double dh = error[dh_index];

    int inputIndex = (inputCh * iHeight * iWidth) + ((fr + r) * iWidth) + (fc + c);
    int filterIndex = (ch * depth * filterHeight * filterWidth) + (inputCh * filterHeight * filterWidth) + (fr * filterWidth) + fc;

    //removed since newErrorGradient isn't really passed on to a higher conv. layer... yet.
    //atomicAdd(&newErrorGradient[inputIndex], filters[filterIndex] * dh);

    //atomicAdd(&delta_filters[filterIndex], originalInput[inputIndex] * dh);
    double val = originalInput[inputIndex] * dh;// / (depth * filterHeight * filterWidth);
//    __syncthreads();
//    if (originalInput[inputIndex] <= 0) {
//        val = 0.0;
//    }

    /**
     * This is supposed to be in the defined function (seen above) as `atomicADD`.
     * Cubix doesn't like the precompiler guards put against it, hpcvis3 likes it.
     * No reversion, inclusion of file or anything else likes to make this work.
     * This code is here provisionally.
     *
     * This WILL be removed in the future
     */
    unsigned long long int *address_as_ull = (unsigned long long int *) &delta_filters[filterIndex];
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}

__global__
void cu_getLength_kernel(double *v, double *lengths) {
    int v_index = blockIdx.x;
    int specificDim = threadIdx.x;
    int dim = blockDim.x;

    extern __shared__
    double shared_v_values[];

    sharedMemoryReduce(shared_v_values, v[v_index*dim+specificDim]*v[v_index*dim+specificDim], specificDim, dim);

    // find out who's the biggest, and save his value in the first number
    if (specificDim == 0) {
        lengths[v_index] = sqrt(shared_v_values[0]);
    }
}

__global__
void cu_getVectorLoss_kernel(double *v, double *truthMap, double *losses, double m_plus, double m_minus, double lambda) {
    int k = blockIdx.x;
    int d = threadIdx.x;
    int dim = blockDim.x;

    extern __shared__
    double shared_vector_lengths[];

    sharedMemoryReduce(shared_vector_lengths, v[k * dim + d] * v[k * dim + d], d, dim);

    if (d == 0) {
        shared_vector_lengths[0] = sqrt(shared_vector_lengths[0] + EPSILON + EPSILON); // length of vector
    }
    __syncthreads();

    double rawMarginLoss = 0.0;
    if (truthMap[k]) {
        rawMarginLoss = max(0.0, m_plus - shared_vector_lengths[0]);
        rawMarginLoss *= rawMarginLoss;
    } else {
        rawMarginLoss = max(0.0, shared_vector_lengths[0] - m_minus);
        rawMarginLoss *= rawMarginLoss;
        rawMarginLoss *= lambda;
    }

    double resizingFactor = rawMarginLoss;
    losses[k] = resizingFactor;
}