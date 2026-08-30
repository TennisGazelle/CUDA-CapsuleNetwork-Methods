#include <MNISTDataIO.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t readBigEndianU32(std::ifstream& input, const std::string& fieldName) {
    unsigned char bytes[4] = {0, 0, 0, 0};
    input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!input) {
        throw std::runtime_error("truncated MNIST file while reading " + fieldName);
    }
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) |
           static_cast<std::uint32_t>(bytes[3]);
}

std::ifstream openBinary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open MNIST file: " + path);
    }
    return input;
}

}  // namespace

std::vector<Image> readMNISTPair(const std::string& imageFile,
                                 const std::string& labelFile) {
    std::ifstream images = openBinary(imageFile);
    std::ifstream labels = openBinary(labelFile);

    const std::uint32_t imageMagic = readBigEndianU32(images, "image magic");
    const std::uint32_t imageCount = readBigEndianU32(images, "image count");
    const std::uint32_t rows = readBigEndianU32(images, "row count");
    const std::uint32_t columns = readBigEndianU32(images, "column count");

    const std::uint32_t labelMagic = readBigEndianU32(labels, "label magic");
    const std::uint32_t labelCount = readBigEndianU32(labels, "label count");

    if (imageMagic != 2051U) {
        throw std::runtime_error("invalid MNIST image magic number");
    }
    if (labelMagic != 2049U) {
        throw std::runtime_error("invalid MNIST label magic number");
    }
    if (imageCount != labelCount) {
        throw std::runtime_error("MNIST image and label counts do not match");
    }
    if (rows != 28U || columns != 28U) {
        throw std::runtime_error("this Capsule Network implementation requires 28x28 MNIST images");
    }

    std::vector<Image> result;
    result.reserve(imageCount);
    std::vector<unsigned char> row(columns);

    for (std::uint32_t imageIndex = 0; imageIndex < imageCount; ++imageIndex) {
        Image image;
        for (std::uint32_t rowIndex = 0; rowIndex < rows; ++rowIndex) {
            images.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
            if (!images) {
                throw std::runtime_error("truncated MNIST image payload");
            }
            image.addRow(row);
        }

        unsigned char label = 0;
        labels.read(reinterpret_cast<char*>(&label), 1);
        if (!labels) {
            throw std::runtime_error("truncated MNIST label payload");
        }
        if (label > 9U) {
            throw std::runtime_error("MNIST label is outside [0, 9]");
        }
        image.setLabel(label);
        result.push_back(image);
    }

    return result;
}
