#include <CapsuleNetwork/BackpropUtils.h>

#include <stdexcept>

void accumulatePrimaryCapsuleError(std::vector<arma::vec>& accumulated,
                                   const std::vector<arma::vec>& contribution) {
    if (accumulated.size() != contribution.size()) {
        throw std::invalid_argument("primary-capsule error contributions must have matching vector counts");
    }

    for (std::size_t i = 0; i < accumulated.size(); ++i) {
        if (accumulated[i].n_elem != contribution[i].n_elem) {
            throw std::invalid_argument("primary-capsule error contributions must have matching vector dimensions");
        }
        accumulated[i] += contribution[i];
    }
}
