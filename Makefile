CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude -Ithird_party/doctest -Itests
LDLIBS ?= -larmadillo

BUILD_DIR := .build
HOST_PRESET ?= ci-host
ASAN_PRESET ?= host-asan
CUDA_PRESET ?= cuda-compile
GPU_PRESET ?= cuda-gpu

TEST_UTILS_BIN := $(BUILD_DIR)/test_utils
TEST_GA_BIN := $(BUILD_DIR)/test_ga
TEST_GA_GENERATION_BIN := $(BUILD_DIR)/test_ga_generation
TEST_BACKPROP_BIN := $(BUILD_DIR)/test_backprop
TEST_MNIST_IO_BIN := $(BUILD_DIR)/test_mnist_io
TEST_EVAL_BIN := $(BUILD_DIR)/test_eval_no_mutation

GA_CORE_SOURCES := \
	src/GA/IndividualCore.cpp \
	src/GA/PopulationCore.cpp \
	src/Utils.cpp \
	src/CapsNetConfig.cpp \
	src/models/PopulationStats.cpp

.PHONY: help configure-host unit-test ctest docs-check package package-check \
	ci cuda-compile gpu-test asan-test format-check clean clean-testing-artifacts

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  configure-host Configure CMake host preset ($(HOST_PRESET))' \
	  '  unit-test      Build and run host-only correctness tests (no CUDA required)' \
	  '  ctest          Run CTest against the host CMake build' \
	  '  docs-check     Validate required docs and relative Markdown links' \
	  '  package        Create versioned source tar.gz + zip + checksums' \
	  '  package-check  Build packages and verify critical files are present' \
	  '  ci             Host PR validation: unit-test + docs-check + package-check' \
	  '  cuda-compile   Configure/build CUDA targets (compile only; needs toolkit)' \
	  '  gpu-test       Run CUDA GPU parity tests (needs GPU)' \
	  '  asan-test      Host tests under AddressSanitizer via CMake preset' \
	  '  format-check   Scoped clang-format dry-run on modernized paths' \
	  '  clean          Remove revival-generated build/release outputs' \
	  '  clean-testing-artifacts Remove testing_artifacts/'

$(BUILD_DIR):
	mkdir -p $@

$(TEST_UTILS_BIN): tests/test_utils.cpp src/Utils.cpp include/Utils.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_utils.cpp src/Utils.cpp -o $@ $(LDLIBS)

$(TEST_GA_BIN): tests/test_ga.cpp $(GA_CORE_SOURCES) include/GA/Individual.h include/GA/Population.h include/models/PopulationStats.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_ga.cpp $(GA_CORE_SOURCES) -o $@ $(LDLIBS)

$(TEST_GA_GENERATION_BIN): tests/test_ga_generation.cpp src/GA/GACore.cpp $(GA_CORE_SOURCES) include/GA/GA.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_ga_generation.cpp src/GA/GACore.cpp $(GA_CORE_SOURCES) -o $@ $(LDLIBS)

$(TEST_BACKPROP_BIN): tests/test_backprop.cpp src/CapsuleNetwork/BackpropUtils.cpp include/CapsuleNetwork/BackpropUtils.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_backprop.cpp src/CapsuleNetwork/BackpropUtils.cpp -o $@ $(LDLIBS)

$(TEST_MNIST_IO_BIN): tests/test_mnist_io.cpp src/MNISTDataIO.cpp src/models/Image.cpp include/MNISTDataIO.h include/models/Image.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_mnist_io.cpp src/MNISTDataIO.cpp src/models/Image.cpp -o $@ $(LDLIBS)

$(TEST_EVAL_BIN): tests/test_eval_no_mutation.cpp src/CapsuleNetwork/EvalPolicy.cpp include/CapsuleNetwork/EvalPolicy.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_eval_no_mutation.cpp src/CapsuleNetwork/EvalPolicy.cpp -o $@

unit-test: $(TEST_UTILS_BIN) $(TEST_GA_BIN) $(TEST_GA_GENERATION_BIN) $(TEST_BACKPROP_BIN) $(TEST_MNIST_IO_BIN) $(TEST_EVAL_BIN)
	$(TEST_UTILS_BIN)
	$(TEST_GA_BIN)
	$(TEST_GA_GENERATION_BIN)
	$(TEST_BACKPROP_BIN)
	$(TEST_MNIST_IO_BIN)
	$(TEST_EVAL_BIN)

configure-host:
	cmake --preset $(HOST_PRESET)

ctest: configure-host
	cmake --build --preset $(HOST_PRESET)
	ctest --preset $(HOST_PRESET) --output-on-failure

docs-check:
	python3 scripts/check_docs.py

package:
	bash scripts/package_release.sh

package-check: package
	@version="$$(tr -d '[:space:]' < VERSION)"; \
	 prefix="cuda-capsule-network-methods-$${version}"; \
	 test -s "release-assets/$${prefix}.tar.gz"; \
	 test -s "release-assets/$${prefix}.zip"; \
	 test -s "release-assets/$${prefix}.sha256"; \
	 tar_listing="$$(mktemp)"; \
	 zip_listing="$$(mktemp)"; \
	 trap 'rm -f "$${tar_listing}" "$${zip_listing}"' EXIT; \
	 tar -tzf "release-assets/$${prefix}.tar.gz" > "$${tar_listing}"; \
	 unzip -l "release-assets/$${prefix}.zip" > "$${zip_listing}"; \
	 grep -q "$${prefix}/README.md" "$${tar_listing}"; \
	 grep -q "$${prefix}/SPEC.md" "$${zip_listing}"; \
	 grep -q "$${prefix}.tar.gz" "release-assets/$${prefix}.sha256"; \
	 grep -q "$${prefix}.zip" "release-assets/$${prefix}.sha256"

ci: unit-test docs-check package-check

cuda-compile:
	cmake --preset $(CUDA_PRESET)
	cmake --build --preset $(CUDA_PRESET)

gpu-test:
	cmake --preset $(GPU_PRESET)
	cmake --build --preset $(GPU_PRESET)
	ctest --preset $(GPU_PRESET) --output-on-failure

asan-test:
	cmake --preset $(ASAN_PRESET)
	cmake --build --preset $(ASAN_PRESET)
	ctest --preset $(ASAN_PRESET) --output-on-failure

format-check:
	bash scripts/check_format.sh

clean:
	rm -rf $(BUILD_DIR) release-assets build

clean-testing-artifacts:
	rm -rf testing_artifacts
