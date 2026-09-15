CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude -Ithird_party/doctest -Itests
LDLIBS ?= -larmadillo

BUILD_DIR := .build

# Preset names match CMakePresets.json
HOST_DEBUG_PRESET := host-debug
HOST_CI_PRESET := host-ci
HOST_ASAN_PRESET := host-asan
CUDA_COMPILE_PRESET := cuda-compile
CUDA_TEST_PRESET := cuda-test

# Test binaries built by the non-CMake path
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

.PHONY: help \
	host-debug host-debug-test host-test host-ci host-asan \
	cuda-check cuda-compile cuda-test \
	docs-check package package-check format-check \
	clean clean-all clean-testing-artifacts

help:
	@printf '%s\n' \
	  'Host targets:' \
	  '  host-debug       Build debug configuration (no tests)' \
	  '  host-debug-test  Run tests against debug build (must exist)' \
	  '  host-test        Build and run host unit tests (quick, no CMake)' \
	  '  host-ci          Full CI bundle: host-test + docs-check + package-check' \
	  '  host-asan        Build and run tests under AddressSanitizer' \
	  '' \
	  'CUDA targets:' \
	  '  cuda-check       Verify CUDA toolkit is available, report missing deps' \
	  '  cuda-compile     Compile CUDA targets (no GPU required, toolkit only)' \
	  '  cuda-test        Build and run CUDA GPU parity tests (requires GPU)' \
	  '' \
	  'Utilities:' \
	  '  docs-check       Validate required docs and relative Markdown links' \
	  '  package          Create versioned source tar.gz + zip + checksums' \
	  '  package-check    Build packages and verify critical files are present' \
	  '  format-check     Scoped clang-format dry-run on modernized paths' \
	  '  clean            Remove build outputs (preserves CMake builds)' \
	  '  clean-all        Remove all build outputs including CMake' \
	  '  clean-testing-artifacts  Remove testing_artifacts/' \
	  '' \
	  'Deferred (not yet implemented):' \
	  '  host-run         Run NeuralNets binary with sample data (host build)' \
	  '  cuda-run         Run NeuralNets binary with sample data (CUDA build)'

# === Build directory ===
$(BUILD_DIR):
	mkdir -p $@

# === Non-CMake test binaries (fast iteration) ===
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

# === Host targets ===
host-debug:
	cmake --preset $(HOST_DEBUG_PRESET)
	cmake --build --preset $(HOST_DEBUG_PRESET)

host-debug-test:
	@if [ ! -d "build/host-debug" ]; then \
		echo "Error: host-debug build does not exist. Run 'make host-debug' first."; \
		exit 1; \
	fi
	ctest --preset $(HOST_DEBUG_PRESET) --output-on-failure

host-test: $(TEST_UTILS_BIN) $(TEST_GA_BIN) $(TEST_GA_GENERATION_BIN) $(TEST_BACKPROP_BIN) $(TEST_MNIST_IO_BIN) $(TEST_EVAL_BIN)
	$(TEST_UTILS_BIN)
	$(TEST_GA_BIN)
	$(TEST_GA_GENERATION_BIN)
	$(TEST_BACKPROP_BIN)
	$(TEST_MNIST_IO_BIN)
	$(TEST_EVAL_BIN)

host-ci: host-test docs-check package-check

host-asan:
	cmake --preset $(HOST_ASAN_PRESET)
	cmake --build --preset $(HOST_ASAN_PRESET)
	ctest --preset $(HOST_ASAN_PRESET) --output-on-failure

# === CUDA targets ===
cuda-check:
	@echo "=== CUDA Environment Check ===" && \
	ok=1; \
	if command -v nvcc >/dev/null 2>&1; then \
		echo "[OK] nvcc: $$(nvcc --version | grep release | head -1)"; \
		nvcc_dir="$$(dirname $$(dirname $$(command -v nvcc)))"; \
		echo "[OK] CUDA path: $$nvcc_dir (from nvcc location)"; \
	else \
		echo "[MISSING] nvcc - install CUDA toolkit:"; \
		echo "         Ubuntu/Debian: apt install nvidia-cuda-toolkit"; \
		echo "         Or download from: https://developer.nvidia.com/cuda-downloads"; \
		ok=0; \
	fi; \
	if command -v nvidia-smi >/dev/null 2>&1; then \
		gpu_name="$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; \
		if [ -n "$$gpu_name" ]; then \
			echo "[OK] GPU: $$gpu_name"; \
		else \
			echo "[INFO] nvidia-smi found but no GPU detected - compile-only mode available"; \
		fi; \
	else \
		echo "[INFO] nvidia-smi not found - compile-only mode available, GPU tests will skip"; \
	fi; \
	if [ $$ok -eq 1 ]; then \
		echo ""; \
		echo "CUDA toolkit ready for compilation."; \
	else \
		echo ""; \
		echo "Missing dependencies for CUDA builds."; \
		exit 1; \
	fi

cuda-compile: cuda-check
	cmake --preset $(CUDA_COMPILE_PRESET)
	cmake --build --preset $(CUDA_COMPILE_PRESET)

cuda-test: cuda-check
	cmake --preset $(CUDA_TEST_PRESET)
	cmake --build --preset $(CUDA_TEST_PRESET)
	ctest --preset $(CUDA_TEST_PRESET) --output-on-failure

# === Utilities ===
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

format-check:
	bash scripts/check_format.sh

clean:
	rm -rf $(BUILD_DIR) release-assets

clean-all: clean
	rm -rf build

clean-testing-artifacts:
	rm -rf testing_artifacts
