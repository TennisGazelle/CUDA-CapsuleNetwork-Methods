//
// Created by Daniel Lopez on 12/29/17.
//

#include <MNISTDataIO.h>
#include <MNISTReader.h>

#include <string>

namespace {

std::string joinPath(const std::string& directory, const std::string& filename) {
    if (directory.empty() || directory == ".") {
        return directory.empty() ? filename : directory + "/" + filename;
    }
    if (directory.back() == '/') {
        return directory + filename;
    }
    return directory + "/" + filename;
}

}  // namespace

MNISTReader::MNISTReader() {
    readMNISTData();
}

MNISTReader* MNISTReader::getInstance() {
    // Function-local static initialization is thread-safe in C++11 and avoids
    // the historical manually locked singleton and unreachable delete path.
    static MNISTReader reader;
    return &reader;
}

std::string MNISTReader::resolveDataDirectory() {
    return resolveMNISTDataDirectory();
}

void MNISTReader::readMNISTData() {
    const std::string directory = resolveDataDirectory();
    trainingData = readMNISTPair(
        joinPath(directory, "train-images-idx3-ubyte"),
        joinPath(directory, "train-labels-idx1-ubyte"));
    testingData = readMNISTPair(
        joinPath(directory, "t10k-images-idx3-ubyte"),
        joinPath(directory, "t10k-labels-idx1-ubyte"));
}

Image MNISTReader::getTrainingImage(int index) const {
    return trainingData.at(static_cast<std::size_t>(index));
}

Image MNISTReader::getTestingImage(int index) const {
    return testingData.at(static_cast<std::size_t>(index));
}

Image* MNISTReader::getTrainingImageRef(int index) {
    return &trainingData.at(static_cast<std::size_t>(index));
}

Image* MNISTReader::getTestingImageRef(int index) {
    return &testingData.at(static_cast<std::size_t>(index));
}
