#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <models/CUUnifiedBlob.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "test_helpers/near.hpp"

namespace {

bool cuda_runtime_available() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

} // namespace

TEST_CASE("cpu_matrix_vector_matches_oracle") {
    // Dimensions: I=2, O=3, K=2, T=2
    const int inputDim = 2;
    const int outputDim = 3;
    const int numClasses = 2;
    const int tensorSize = 2;
    const int elements = numClasses * tensorSize;
    const int matrixSize = inputDim * outputDim * elements;
    const int inputSize = inputDim * elements;
    const int outputSize = outputDim * elements;

    CUUnifiedBlob matrix(matrixSize);
    CUUnifiedBlob input(inputSize);
    CUUnifiedBlob output(outputSize);
    matrix.fillSequentially();
    input.fillSequentially();
    output.clear();

    CUUnifiedBlob::matrixVectorMultiplication(matrix, input, output, inputDim, outputDim,
                                              numClasses, tensorSize);

    std::vector<double> matrixHost(static_cast<std::size_t>(matrixSize));
    std::vector<double> inputHost(static_cast<std::size_t>(inputSize));
    std::vector<double> oracle(static_cast<std::size_t>(outputSize), 0.0);
    for (int i = 0; i < matrixSize; ++i) {
        matrixHost[static_cast<std::size_t>(i)] = matrix.getValueAt_1D(i);
    }
    for (int i = 0; i < inputSize; ++i) {
        inputHost[static_cast<std::size_t>(i)] = input.getValueAt_1D(i);
    }
    capsnet::test::matrix_vector_mult_oracle(matrixHost, inputHost, oracle, inputDim, outputDim,
                                             numClasses, tensorSize);

    for (int i = 0; i < outputSize; ++i) {
        CHECK((capsnet::test::near(output.getValueAt_1D(i), oracle[static_cast<std::size_t>(i)],
                                   1e-9)));
    }
}

TEST_CASE("cuda_matrix_vector_matches_oracle") {
    if (!cuda_runtime_available()) {
        MESSAGE("skipping CUDA matrix-vector parity: no GPU runtime");
        return;
    }

    const int inputDim = 2;
    const int outputDim = 3;
    const int numClasses = 2;
    const int tensorSize = 2;
    const int elements = numClasses * tensorSize;
    const int matrixSize = inputDim * outputDim * elements;
    const int inputSize = inputDim * elements;
    const int outputSize = outputDim * elements;

    CUUnifiedBlob matrix(matrixSize);
    CUUnifiedBlob input(inputSize);
    CUUnifiedBlob outputCpu(outputSize);
    CUUnifiedBlob outputCuda(outputSize);
    matrix.fillSequentially();
    input.fillSequentially();
    outputCpu.clear();
    outputCuda.CUDA_clear();

    CUUnifiedBlob::matrixVectorMultiplication(matrix, input, outputCpu, inputDim, outputDim,
                                              numClasses, tensorSize);
    CUUnifiedBlob::CUDA_matrixVectorMultiplication(matrix, input, outputCuda, inputDim, outputDim,
                                                   numClasses, tensorSize);

    for (int i = 0; i < outputSize; ++i) {
        CHECK((capsnet::test::near(outputCpu.getValueAt_1D(i), outputCuda.getValueAt_1D(i), 1e-8)));
    }
}
