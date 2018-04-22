//
// Created by daniellopez on 4/4/18.
//

#include <cassert>
#include <CUDAUtils.h>
#include <iostream>
#include <cmath>
#include "models/CUUnifiedBlob.h"
#include "CUDAUtils.h"

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

void CUUnifiedBlob::print(const std::string& msg, int width) {
    if (!msg.empty()) {
        std::cout << msg << std::endl;
    }
    int bufferSize = std::min(size, 200);
    for (int i = 0; i < bufferSize; i++) {
        std::cout << data[i] << "\t";
        if (((i+1) % width) == 0) {
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
            std::cout << "they didn't match at: " << i << std::endl;
            std::cout << "this: " << data[i] << " other: " << other.data[i] << std::endl;
            return false;
        }
    }

    return true;
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
            outputVector.data[i] += inputVector.data[j] * matrix.data[i*inputDim + j];
        }
    }
}

void CUUnifiedBlob::vectorVectorSoftmax(CUUnifiedBlob& b, CUUnifiedBlob& c,
                                        int numClasses, int tensorSize) {
    for (int k = 0; k < numClasses; k++) {
        double sum_b_exps = 0.0;
        for (int t = 0; t < tensorSize; t++) {
            sum_b_exps += exp(b.data[t*numClasses + k]);
        }

        // then go through the c's and set accordingly
        for (int t = 0; t < tensorSize; t++) {
            c.data[t*numClasses + k] = exp(b.data[t*numClasses + k])/ sum_b_exps;
        }
    }
}

void CUUnifiedBlob::weightReduceVectors(CUUnifiedBlob &u_hat, CUUnifiedBlob &c, CUUnifiedBlob &v, int numClasses,
                                        int tensorSize, int dim) {
    for (int k = 0; k < numClasses; k++) {
        for (int t = 0; t < tensorSize; t++) {
            int u_hat_index = t*numClasses*dim + k*dim;

            for (int i = u_hat_index; i < u_hat_index + dim; i++) {
                v.data[i % (numClasses*dim)] += u_hat.data[i] * c.data[t*numClasses+k];
            }
        }
    }
}

void CUUnifiedBlob::vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim) {
    for (int v_index = 0; v_index < numVecs*vecDim; v_index += vecDim) {
        double sum_squares = 0;
    	for (int i = 0; i < vecDim; i++) {
    	    sum_squares += pow(v.data[v_index + i], 2);
    	}
    	double squashFactor = sum_squares / (1.0 + sum_squares);
    	sum_squares = sqrt(sum_squares);
    	for (int i = 0; i < vecDim; i++) {
    		v.data[v_index + i] *= squashFactor / sum_squares;
    	}
    }
}

void CUUnifiedBlob::vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim) {
    for (int k = 0; k < numClasses; k++) {
        int v_index = k*dim;
    	for (int t = 0; t < tensorSize; t++) {
    		int u_hat_index = t*numClasses*dim + k*dim;
    		int b_index = t*numClasses + k;

    		for (int i = 0; i < dim; i++) {
    			b.data[b_index] += u_hat.data[u_hat_index + i] * v.data[v_index + i];
    		}
    	}
    }
}

void CUUnifiedBlob::vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim) {
    const double m_plus = 0.9;
    const double m_minus = 0.1;
    const double lambda = 0.5;

    for (int classIndex = 0; classIndex < numClasses; classIndex++) {
        double sumOfSquaredValues = 0.0;
        for (int i = 0; i < dim; i++) {
            sumOfSquaredValues += std::pow(v.data[classIndex*dim + i], 2);
        }
        double vec_length = sqrt(sumOfSquaredValues + 1e-4);

        double activationFactor = 2*vec_length / pow((vec_length * vec_length) + 1, 2);

        double errorGradient;
        if (vec_length < m_plus) {
            if (vec_length <= m_minus) {
                errorGradient = -2 * truthMap.data[classIndex] * (m_plus - vec_length);
            } else {
                errorGradient = 2 * ((lambda * (truthMap.data[classIndex] - 1) * (m_minus - vec_length)) + truthMap.data[classIndex] * (vec_length - m_plus));
            }
        } else {
            errorGradient = 2 * lambda * (truthMap.data[classIndex] - 1) * (m_minus - vec_length);
        }

        double rawMarginLoss;
        if (truthMap.data[classIndex]) {
            rawMarginLoss = pow(std::max(0.0, m_plus - vec_length), 2);
        } else {
            rawMarginLoss = lambda * pow(std::max(0.0, vec_length - m_minus), 2);
        }

        double resizingFactor = activationFactor * errorGradient * rawMarginLoss / vec_length;
        for (int i = 0; i < dim; i++) {
            v.data[classIndex*dim + i] *= resizingFactor;
        }
    }
}

void CUUnifiedBlob::weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &c, CUUnifiedBlob &w,
                                               CUUnifiedBlob &v_error, int numClasses,
                                               int tensorSize, int innerDim, int outerDim) {
    for (int t = 0; t < tensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            int c_index = t*numClasses + k;
            int w_index = c_index * innerDim * outerDim;
            int v_index = k * outerDim;
            int u_index = c_index * innerDim;
            // think transposed matrix
            for (int col = 0; col < innerDim; col++) {
                double u_value = 0.0;
                for (int row = 0; row < outerDim; row++) {
                    u_value += w.data[row*innerDim + col + w_index] * v_error.data[row + v_index];
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
            int element_index = t*numClasses + k;
            int w_index = element_index * innerDim * outerDim;
            int v_index = k * outerDim;
            int u_index = element_index * innerDim;

            for (int row = 0; row < outerDim; row++) {
                for (int col = 0; col < innerDim; col++) {
                    w.data[row*innerDim + col + w_index] += v_error.data[row + v_index] * old_u.data[col + u_index];
                }
            }
        }
    }
}

void CUUnifiedBlob::multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim) {
	for (int t = 0; t < tensorSize; t++) {
		for (int d = 0; d < dim; d++) {
			for (int k = 1; k < numClasses; k++) {
				int element_index = t*numClasses + k;
				u.data[t*numClasses*dim + d] += u.data[element_index*dim + d];
				u.data[element_index*dim + d] = 0;
			}
		}
	}
}

void CUUnifiedBlob::matrixMatrixUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_error, int size) {
	for (int i = 0; i < size; i++) {
		w.data[i] += w_error.data[i];
		w_error.data[i] = 0;
	}
}

void CUUnifiedBlob::CUDA_matrixVectorMultiplication(CUUnifiedBlob &matrix,
                                                    CUUnifiedBlob &inputVector,
                                                    CUUnifiedBlob &outputVector,
                                                    int inputDim,
                                                    int outputDim,
                                                    int numMultiplications) {
    cu_matrixVectorMultiplication_kernel<<<numMultiplications, outputDim>>>(matrix.data,
                                                    inputVector.data,
                                                    outputVector.data,
                                                    inputDim,
                                                    outputDim);
}

void CUUnifiedBlob::CUDA_vectorVectorSoftmax(CUUnifiedBlob &b,
                                             CUUnifiedBlob &c,
                                             int numClasses,
                                             int tensorSize) {
    int offset = 0;
    int numThreads = std::min(1024, tensorSize);
    cu_vectorVectorSoftmax_kernel<<<numClasses, numThreads, numThreads*sizeof(double)>>>(b.data, c.data, numClasses, tensorSize);
}

void CUUnifiedBlob::CUDA_weightReduceVectors(CUUnifiedBlob &u_hat,
                                             CUUnifiedBlob &c,
                                             CUUnifiedBlob &v,
                                             int numClasses,
                                             int tensorSize,
                                             int dim) {
    dim3 blockDimensions(numClasses, dim);
    int numThreads = std::min(1024, tensorSize);
    cu_weightReduceVector_kernel<<<blockDimensions, numThreads, numThreads*sizeof(double)>>>(u_hat.data, c.data, v.data, numClasses, tensorSize, dim);
}

void CUUnifiedBlob::CUDA_vectorSquash(CUUnifiedBlob &v, int numVecs, int vecDim) {
    cu_vectorSquash_kernel<<<numVecs, vecDim, vecDim*sizeof(double)>>>(v.data, numVecs, vecDim);
}

void CUUnifiedBlob::CUDA_vectorVectorScalarProduct(CUUnifiedBlob &u_hat, CUUnifiedBlob &v, CUUnifiedBlob &b, int numClasses, int tensorSize, int dim) {
    dim3 blockDims(tensorSize, numClasses);
    cu_vectorVectorScalarProduct_kernel<<<blockDims, dim, dim*sizeof(double)>>>(u_hat.data, v.data, b.data, numClasses, tensorSize, dim);
}

void CUUnifiedBlob::CUDA_vectorLossFunction(CUUnifiedBlob &v, CUUnifiedBlob &truthMap, int numClasses, int dim) {
    cu_vectorLossFunction_kernel<<<numClasses, dim, dim*sizeof(double)>>>(v.data, truthMap.data);
}

void CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(CUUnifiedBlob &delta_u, CUUnifiedBlob &c, CUUnifiedBlob &w,
                                                    CUUnifiedBlob &v_error, int numClasses, int tensorSize,
                                                    int innerDim, int outerDim) {
    dim3 blockDims(tensorSize, numClasses);
    cu_weightedTransMatrixVecMult_kernel<<<blockDims, innerDim, innerDim*sizeof(double)>>>(delta_u.data, c.data, w.data, v_error.data, innerDim, outerDim);
}

void CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(CUUnifiedBlob &w, CUUnifiedBlob &v_error, CUUnifiedBlob &old_u, int numClasses, int tensorSize, int innerDim, int outerDim) {
	dim3 blockDims(tensorSize, numClasses);
	dim3 threadDims(outerDim, innerDim);
	cu_vectorVectorMatrixProductAndSum_kernel<<<blockDims,threadDims>>>(w.data, v_error.data, old_u.data, numClasses, tensorSize, innerDim, outerDim);
}

void CUUnifiedBlob::CUDA_multiVectorReduction(CUUnifiedBlob &u, int numClasses, int tensorSize, int dim) {
    dim3 blockDims(tensorSize);
    cu_multiVectorReduction_kernel<<<blockDims, dim>>>(u.data, numClasses, dim);
}

void CUUnifiedBlob::CUDA_matrixMatrixUpdate(CUUnifiedBlob &w, CUUnifiedBlob &w_update, int size) {
    cu_matrixMatrixUpdate_kernel<<<size, 1>>>(w.data, w_update.data);	
}

__global__
void cu_matrixVectorMultiplication_kernel(double *matrix, double *inputVector, double *outputVector,
                                          int inputDim, int outputDim) {
    int u_hat_index = threadIdx.x + (blockIdx.x * outputDim);
    double cache = 0.0;
    for (int c = 0; c < inputDim; c++) {
        cache += matrix[u_hat_index*inputDim+c] * inputVector[blockIdx.x*inputDim+c];
    }
    outputVector[u_hat_index] = cache;
}

__global__
void cu_vectorVectorSoftmax_kernel(double *b, double *c, int numClasses, int tensorSize) {
    int t = threadIdx.x;
    int k = blockIdx.x;

    extern __shared__
    double shared_b_exps[];

    double my_exp_bs [2]; // make this dynamic and only as needed
    for (int i = 0; i*1024 < tensorSize; i++) {
        if (i*1024 + t < tensorSize) {
            my_exp_bs[i] = exp(b[(i*1024+t)*numClasses+k]); // consider using hexp() for speed
//            if (!isnan(my_exp_bs[i])) {
//                my_exp_bs[i] = 0;
//            }
            shared_b_exps[t] += my_exp_bs[i];
        }
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (t % (2*s) == 0 && !isinf(shared_b_exps[t+s] + shared_b_exps[t])) {
            shared_b_exps[t] += shared_b_exps[t+s];
        }
        __syncthreads();
    }

    double sum_exps = shared_b_exps[0];
    for (int i = 0; i*1024 < tensorSize; i++) {
        if (i*1024 + t < tensorSize) {
            c[(i*1024+t)*numClasses+k] = my_exp_bs[i] / sum_exps;
        }
    }
}

__global__
void cu_weightReduceVector_kernel(double *u_hat, double *c, double *v, int numClasses, int tensorSize, int dim) {
    int k = blockIdx.x;
    int specificDim = blockIdx.y;
    int t = threadIdx.x;

    int u_hat_index = t*numClasses*dim + k*dim;
    int c_index = t*numClasses+k;
    extern __shared__
    double shared_v_vec[];

    shared_v_vec[t] = u_hat[u_hat_index + specificDim] * c[c_index];
    // if tensorsize > 1024, add them to the shared mem as well
    for (int i = 1; i*1024 < tensorSize; i++) {
        int u_hat_offset = 1024*numClasses*dim;
        int c_offset = 1024*numClasses;
        if (i*1024 + t < tensorSize) {
            shared_v_vec[t] += u_hat[u_hat_index + specificDim + u_hat_offset] * c[c_index + c_offset];
        }
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (t % (2*s) == 0) {
            shared_v_vec[t] += shared_v_vec[t + s];
        }
        __syncthreads();
    }

    if (t == 0) {
        v[k*dim + specificDim] = shared_v_vec[0];
    }
}

__global__
void cu_vectorSquash_kernel(double *v, int numVecs, int vecDim) {
    int v_index = blockIdx.y*gridDim.x + blockIdx.x;
    int v_val_index = threadIdx.x;

    extern __shared__
    double shared_v_values[];
    // reduce the square of the individual elements in shared mem
    if (v_index < numVecs) {
    	shared_v_values[v_val_index] = pow(v[v_index*vecDim + v_val_index], 2);
    }
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (v_val_index % (2*s) == 0) {
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

    if (v_val_index == 0 && v_index < numVecs) {
        v[v_index*vecDim + v_val_index] *= shared_v_values[1] / shared_v_values[0];
    }
}

__global__
void cu_vectorVectorScalarProduct_kernel(double *u_hat, double *v, double *b, int numClasses, int tensorSize, int dim) {
    int k = blockIdx.y;
    int specificDim = threadIdx.x;
    int t = blockIdx.x;
    int u_hat_index = t*numClasses*dim + k*dim;

    extern __shared__
    double shared_scalar_products[];
    shared_scalar_products[specificDim] = u_hat[u_hat_index + specificDim] * v[k*dim + specificDim];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s*= 2) {
    	if (specificDim % (2*s) == 0) {
    		shared_scalar_products[specificDim] += shared_scalar_products[specificDim + s];
    	}
    	__syncthreads();
    }

    if (specificDim == 0 && !isnan(shared_scalar_products[0])) {
        b[t*numClasses+k] += shared_scalar_products[0];
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
    shared_vector_lengths[d] = v[k*dim + d] * v[k*dim + d];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (d % (2*s) == 0) {
            shared_vector_lengths[d] += shared_vector_lengths[d+s];
        }
        __syncthreads();
    }
    if (d == 0) {
        shared_vector_lengths[0] = sqrt(shared_vector_lengths[0] + 1e-4);
    }
    __syncthreads();

    double activationFactor = 2*shared_vector_lengths[0] / pow(shared_vector_lengths[0]*shared_vector_lengths[0]+1, 2);

    double errorGradient;
    if (shared_vector_lengths[0] < m_plus) {
        if (shared_vector_lengths[0] <= m_minus) {
            errorGradient = -2 * truthMap[k] * (m_plus - shared_vector_lengths[0]);
        } else {
            errorGradient = 2 * ((lambda * (truthMap[k] - 1) * (m_minus - shared_vector_lengths[0])) + truthMap[k] * (shared_vector_lengths[0] - m_plus));
        }
    } else {
        errorGradient = 2 * lambda * (truthMap[k] - 1) * (m_minus - shared_vector_lengths[0]);
    }

    double rawMarginLoss;
    if (truthMap[k]) {
        rawMarginLoss = pow(max(0.0, m_plus - shared_vector_lengths[0]), 2);
    } else {
        rawMarginLoss = lambda * pow(max(0.0, shared_vector_lengths[0] - m_minus), 2);
    }

    double resizingFactor = activationFactor * errorGradient * rawMarginLoss / shared_vector_lengths[0];
    v[k*dim + d] *= resizingFactor;
}

__global__
void cu_weightedTransMatrixVecMult_kernel(double *delta_u, double *c, double *w, double *v_error, int innerDim, int outerDim) {
    int t = blockIdx.x;
    int k = blockIdx.y;

    int c_index = t*gridDim.y + k;
    int w_index = c_index * innerDim * outerDim;
    int u_index = c_index * innerDim;
    int v_index = k * outerDim;

    int col = threadIdx.x;

    double u_value = 0.0;
    for (int row = 0; row < outerDim; row++) {
        u_value += w[row*innerDim + col + w_index] * v_error[row + v_index];
    }
    delta_u[col + u_index] = u_value * c[c_index];
}

__global__
void cu_vectorVectorMatrixProductAndSum_kernel(double *w, double *v_error, double *old_u, int numClasses, int tensorSize, int innerDim, int outerDim) {
    int t = blockIdx.x;
    int k = blockIdx.y;

    int row = threadIdx.x;
    int col = threadIdx.y;

    int element_index = t*numClasses + k;
    int w_index = element_index * innerDim * outerDim;
    int u_index = element_index * innerDim;
    int v_index = k * outerDim;

	w[row*innerDim + col + w_index] += v_error[row + v_index] * old_u[col + u_index];
}

__global__
void cu_multiVectorReduction_kernel(double *u, int numClasses, int dim) {
	int t = blockIdx.x;
	int d = threadIdx.x;

    for (int i = 1; i < numClasses; i++) {
        int u_element_index = t*numClasses + i;
    	u[t*numClasses*dim + d] += u[u_element_index*dim + d];
    	u[u_element_index*dim + d] = 0;
    }
}

__global__
void cu_matrixMatrixUpdate_kernel(double *w, double *w_error) {
    int tid = blockIdx.x;
    w[tid] += w_error[tid];
    w_error[tid] = 0;
}