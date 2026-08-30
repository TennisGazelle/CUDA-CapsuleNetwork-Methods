#ifndef NEURALNETS_BACKPROPUTILS_H
#define NEURALNETS_BACKPROPUTILS_H

#include <armadillo>
#include <vector>

// Add one upper-capsule backpropagation contribution into the accumulated
// error for the flattened primary-capsule vectors. Both tensors must have the
// same number of vectors and matching vector dimensions.
void accumulatePrimaryCapsuleError(std::vector<arma::vec>& accumulated,
                                   const std::vector<arma::vec>& contribution);

#endif //NEURALNETS_BACKPROPUTILS_H
