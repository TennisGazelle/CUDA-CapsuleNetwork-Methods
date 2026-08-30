#include <CapsuleNetwork/BackpropUtils.h>

#include <stdexcept>
#include <string>

void validateDigitCapsuleErrors(const std::vector<arma::vec>& errors,
                                std::size_t expectedCount,
                                arma::uword expectedDimension) {
    if (errors.size() != expectedCount) {
        throw std::invalid_argument("digit-capsule error count does not match output capsule count");
    }

    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (errors[i].n_elem != expectedDimension) {
            throw std::invalid_argument(
                "digit-capsule error vector " + std::to_string(i) +
                " dimension does not match configuration");
        }
    }
}

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
