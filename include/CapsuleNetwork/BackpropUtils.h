#ifndef NEURALNETS_BACKPROPUTILS_H
#define NEURALNETS_BACKPROPUTILS_H

#include <armadillo>
#include <cstddef>
#include <vector>

// Validate the outer error tensor before any capsule mutates its accumulated
// gradients. Every digit capsule must receive one vector of the configured
// output dimension.
void validateDigitCapsuleErrors(const std::vector<arma::vec>& errors,
                                std::size_t expectedCount,
                                arma::uword expectedDimension);

// Add one upper-capsule backpropagation contribution into the accumulated
// error for the flattened primary-capsule vectors. Both tensors must have the
// same number of vectors and matching vector dimensions.
void accumulatePrimaryCapsuleError(std::vector<arma::vec>& accumulated,
                                   const std::vector<arma::vec>& contribution);

#endif //NEURALNETS_BACKPROPUTILS_H
