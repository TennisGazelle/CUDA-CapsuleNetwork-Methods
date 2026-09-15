#pragma once

#include <Utils.h>

namespace capsnet {
namespace test {

inline void set_utils_seed(unsigned seed) {
    Utils::setRandomSeed(seed);
}

} // namespace test
} // namespace capsnet
