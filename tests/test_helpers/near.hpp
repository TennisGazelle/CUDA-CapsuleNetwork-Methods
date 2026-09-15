#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace capsnet {
namespace test {

inline bool near(double lhs, double rhs, double tolerance = 1e-10) {
    return std::abs(lhs - rhs) <= tolerance;
}

/// CPU oracle for CUUnifiedBlob::matrixVectorMultiplication layout.
/// Layout: for each (t,k), input is length inputDim, matrix is outputDim x inputDim
/// row-major, output is length outputDim. See docs/CUDA_ARCHITECTURE.md §3.2.
inline void matrix_vector_mult_oracle(const std::vector<double>& matrix,
                                      const std::vector<double>& input, std::vector<double>& output,
                                      int inputDim, int outputDim, int numClasses, int tensorSize) {
    const std::size_t expectedOut = static_cast<std::size_t>(outputDim) *
                                    static_cast<std::size_t>(numClasses) *
                                    static_cast<std::size_t>(tensorSize);
    if (output.size() != expectedOut) {
        throw std::invalid_argument("matrix_vector_mult_oracle: output size mismatch");
    }
    std::fill(output.begin(), output.end(), 0.0);
    for (int t = 0; t < tensorSize; ++t) {
        for (int k = 0; k < numClasses; ++k) {
            const int elementIndex = t * numClasses + k;
            const int inputIndex = elementIndex * inputDim;
            const int outputIndex = elementIndex * outputDim;
            const int matrixIndex = elementIndex * inputDim * outputDim;
            for (int i = 0; i < outputDim; ++i) {
                for (int j = 0; j < inputDim; ++j) {
                    const double cell =
                        input[static_cast<std::size_t>(j + inputIndex)] *
                        matrix[static_cast<std::size_t>((i * inputDim + j) + matrixIndex)];
                    if (!std::isnan(cell)) {
                        output[static_cast<std::size_t>(i + outputIndex)] += cell;
                    }
                }
            }
        }
    }
}

inline bool vectors_near(const std::vector<double>& lhs, const std::vector<double>& rhs,
                         double tolerance = 1e-8) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!near(lhs[i], rhs[i], tolerance)) {
            return false;
        }
    }
    return true;
}

} // namespace test
} // namespace capsnet
