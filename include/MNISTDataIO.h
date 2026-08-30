#ifndef NEURALNETS_MNISTDATAIO_H
#define NEURALNETS_MNISTDATAIO_H

#include <models/Image.h>

#include <string>
#include <vector>

std::vector<Image> readMNISTPair(const std::string& imageFile,
                                 const std::string& labelFile);

#endif //NEURALNETS_MNISTDATAIO_H
