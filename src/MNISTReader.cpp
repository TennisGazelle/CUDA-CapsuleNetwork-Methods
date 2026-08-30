//
// Created by Daniel Lopez on 12/29/17.
//

#include <MNISTDataIO.h>
#include <MNISTReader.h>

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

bool fileExists(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

bool containsMNIST(const std::string& directory) {
    return fileExists(joinPath(directory, "train-images-idx3-ubyte")) &&
           fileExists(joinPath(directory, "train-labels-idx1-ubyte")) &&
           fileExists(joinPath(directory, "t10k-images-idx3-ubyte")) &&
           fileExists(joinPath(directory, "t10k-labels-idx1-ubyte"));
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
    const char* configured = std::getenv("CAPSNET_DATA_DIR");
    if (configured != nullptr && configured[0] != '\0') {
        const std::string directory(configured);
        if (!containsMNIST(directory)) {
            throw std::runtime_error("CAPSNET_DATA_DIR does not contain the four required MNIST IDX files: " + directory);
        }
        return directory;
    }

    const std::vector<std::string> candidates = {"data", "../data"};
    for (const std::string& directory : candidates) {
        if (containsMNIST(directory)) {
            return directory;
        }
    }

    throw std::runtime_error(
        "MNIST data not found. Run from the repository/build directory or set CAPSNET_DATA_DIR explicitly.");
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
