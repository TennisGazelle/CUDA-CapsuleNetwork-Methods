#include <MNISTDataIO.h>

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace {

const std::string kImages = ".build/test-mnist-images.idx3";
const std::string kLabels = ".build/test-mnist-labels.idx1";
const std::string kDataDirectory = ".build/test-mnist-data";
const std::vector<std::string> kDataFilenames = {
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
};

void ensureFixtureDirectory() {
    // CTest/asan paths do not create `.build/` the way `make unit-test` does.
    // Create it (and ignore EEXIST) before writing fixture files.
    if (mkdir(".build", 0700) != 0 && errno != EEXIST) {
        throw std::runtime_error("unable to create .build fixture directory");
    }
}

void writeU32(std::ofstream& output, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((value >> 24U) & 0xffU),
        static_cast<unsigned char>((value >> 16U) & 0xffU),
        static_cast<unsigned char>((value >> 8U) & 0xffU),
        static_cast<unsigned char>(value & 0xffU),
    };
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void writeDataset(std::uint32_t imageMagic = 2051U,
                  std::uint32_t imageCount = 2U,
                  std::uint32_t labelMagic = 2049U,
                  std::uint32_t labelCount = 2U,
                  bool truncateImages = false,
                  std::uint32_t rows = 28U,
                  std::uint32_t columns = 28U,
                  bool truncateLabels = false,
                  unsigned char firstLabel = 0U) {
    ensureFixtureDirectory();
    std::ofstream images(kImages, std::ios::binary | std::ios::trunc);
    writeU32(images, imageMagic);
    writeU32(images, imageCount);
    writeU32(images, rows);
    writeU32(images, columns);

    const std::size_t bytes = static_cast<std::size_t>(imageCount) * rows * columns;
    const std::size_t actualBytes = truncateImages && bytes > 0 ? bytes - 1 : bytes;
    for (std::size_t i = 0; i < actualBytes; ++i) {
        const unsigned char pixel = static_cast<unsigned char>(i % 251U);
        images.write(reinterpret_cast<const char*>(&pixel), 1);
    }

    std::ofstream labels(kLabels, std::ios::binary | std::ios::trunc);
    writeU32(labels, labelMagic);
    writeU32(labels, labelCount);
    const std::uint32_t labelsToWrite = truncateLabels && labelCount > 0 ? labelCount - 1 : labelCount;
    for (std::uint32_t i = 0; i < labelsToWrite; ++i) {
        const unsigned char label = i == 0 ? firstLabel : static_cast<unsigned char>(i % 10U);
        labels.write(reinterpret_cast<const char*>(&label), 1);
    }
}

template <typename Fn>
bool throwsRuntimeError(Fn fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_valid_pair() {
    writeDataset();
    const std::vector<Image> images = readMNISTPair(kImages, kLabels);
    assert(images.size() == 2);
    assert(images[0].size() == 28U * 28U);
    assert(images[0].getLabel() == 0);
    assert(images[1].getLabel() == 1);
    assert(images[0][0] == 0);
    assert(images[0][1] == 1);
}

void test_bad_magic_is_rejected() {
    writeDataset(1234U);
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));

    writeDataset(2051U, 2U, 1234U);
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));
}

void test_count_mismatch_is_rejected() {
    writeDataset(2051U, 2U, 2049U, 1U);
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));
}

void test_truncated_payload_is_rejected() {
    writeDataset(2051U, 2U, 2049U, 2U, true);
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));
}

void test_missing_file_is_rejected() {
    assert(throwsRuntimeError([] { readMNISTPair(".build/does-not-exist", kLabels); }));
}

void test_dimensions_and_labels_are_validated() {
    writeDataset(2051U, 2U, 2049U, 2U, false, 27U, 28U);
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));

    writeDataset(2051U, 2U, 2049U, 2U, false, 28U, 28U, false, 10U);
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));
}

void test_truncated_header_and_label_payload_are_rejected() {
    ensureFixtureDirectory();
    std::ofstream images(kImages, std::ios::binary | std::ios::trunc);
    writeU32(images, 2051U);
    images.close();
    std::ofstream labels(kLabels, std::ios::binary | std::ios::trunc);
    writeU32(labels, 2049U);
    writeU32(labels, 0U);
    labels.close();
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));

    writeDataset(2051U, 2U, 2049U, 2U, false, 28U, 28U, true);
    assert(throwsRuntimeError([] { readMNISTPair(kImages, kLabels); }));
}

void test_zero_count_dataset_is_accepted() {
    writeDataset(2051U, 0U, 2049U, 0U);
    assert(readMNISTPair(kImages, kLabels).empty());
}

void test_explicit_data_directory_resolution() {
    ensureFixtureDirectory();
    assert(mkdir(kDataDirectory.c_str(), 0700) == 0 || errno == EEXIST);
    for (const std::string& filename : kDataFilenames) {
        std::ofstream(kDataDirectory + "/" + filename, std::ios::binary).close();
    }

    assert(setenv("CAPSNET_DATA_DIR", kDataDirectory.c_str(), 1) == 0);
    assert(resolveMNISTDataDirectory() == kDataDirectory);
    assert(setenv("CAPSNET_DATA_DIR", ".build/missing-mnist-data", 1) == 0);
    assert(throwsRuntimeError([] { resolveMNISTDataDirectory(); }));
    assert(unsetenv("CAPSNET_DATA_DIR") == 0);
}

void cleanup() {
    std::remove(kImages.c_str());
    std::remove(kLabels.c_str());
    for (const std::string& filename : kDataFilenames) {
        std::remove((kDataDirectory + "/" + filename).c_str());
    }
    assert(rmdir(kDataDirectory.c_str()) == 0);
}

}  // namespace

int main() {
    test_valid_pair();
    test_bad_magic_is_rejected();
    test_count_mismatch_is_rejected();
    test_truncated_payload_is_rejected();
    test_missing_file_is_rejected();
    test_dimensions_and_labels_are_validated();
    test_truncated_header_and_label_payload_are_rejected();
    test_zero_count_dataset_is_accepted();
    test_explicit_data_directory_resolution();
    cleanup();
    std::cout << "host MNIST IDX parser tests passed" << std::endl;
    return 0;
}
