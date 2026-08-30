#ifndef NEURALNETS_MNISTDATAIO_H
#define NEURALNETS_MNISTDATAIO_H

#include <models/Image.h>

#include <string>
#include <vector>

std::vector<Image> readMNISTPair(const std::string& imageFile,
                                 const std::string& labelFile);

// Resolve the four-file MNIST dataset directory from CAPSNET_DATA_DIR or the
// repository/build-directory fallbacks. Exposed separately for host testing.
std::string resolveMNISTDataDirectory();

#endif //NEURALNETS_MNISTDATAIO_H
