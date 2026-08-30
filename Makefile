CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude
LDLIBS ?= -larmadillo

BUILD_DIR := .build
TEST_UTILS_BIN := $(BUILD_DIR)/test_utils
TEST_GA_BIN := $(BUILD_DIR)/test_ga
TEST_GA_GENERATION_BIN := $(BUILD_DIR)/test_ga_generation

GA_CORE_SOURCES := \
	src/GA/IndividualCore.cpp \
	src/GA/PopulationCore.cpp \
	src/Utils.cpp \
	src/CapsNetConfig.cpp \
	src/models/PopulationStats.cpp

.PHONY: help unit-test docs-check package package-check ci clean

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  unit-test     Build and run host-only correctness tests (no CUDA required)' \
	  '  docs-check    Validate required docs and relative Markdown links' \
	  '  package       Create versioned source tar.gz + zip under release-assets/' \
	  '  package-check Build packages and verify critical files are present' \
	  '  ci            Host PR validation bundle: unit-test + docs-check + package-check' \
	  '  clean         Remove revival-generated build/release outputs'

$(BUILD_DIR):
	mkdir -p $@

$(TEST_UTILS_BIN): tests/test_utils.cpp src/Utils.cpp include/Utils.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_utils.cpp src/Utils.cpp -o $@ $(LDLIBS)

$(TEST_GA_BIN): tests/test_ga.cpp $(GA_CORE_SOURCES) include/GA/Individual.h include/GA/Population.h include/models/PopulationStats.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_ga.cpp $(GA_CORE_SOURCES) -o $@ $(LDLIBS)

$(TEST_GA_GENERATION_BIN): tests/test_ga_generation.cpp src/GA/GACore.cpp $(GA_CORE_SOURCES) include/GA/GA.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_ga_generation.cpp src/GA/GACore.cpp $(GA_CORE_SOURCES) -o $@ $(LDLIBS)

unit-test: $(TEST_UTILS_BIN) $(TEST_GA_BIN) $(TEST_GA_GENERATION_BIN)
	$(TEST_UTILS_BIN)
	$(TEST_GA_BIN)
	$(TEST_GA_GENERATION_BIN)

docs-check:
	python3 scripts/check_docs.py

package:
	bash scripts/package_release.sh

package-check: package
	@version="$$(tr -d '[:space:]' < VERSION)"; \
	 prefix="cuda-capsule-network-methods-$${version}"; \
	 test -s "release-assets/$${prefix}.tar.gz"; \
	 test -s "release-assets/$${prefix}.zip"; \
	 tar_listing="$$(mktemp)"; \
	 zip_listing="$$(mktemp)"; \
	 trap 'rm -f "$${tar_listing}" "$${zip_listing}"' EXIT; \
	 tar -tzf "release-assets/$${prefix}.tar.gz" > "$${tar_listing}"; \
	 unzip -l "release-assets/$${prefix}.zip" > "$${zip_listing}"; \
	 grep -q "$${prefix}/README.md" "$${tar_listing}"; \
	 grep -q "$${prefix}/SPEC.md" "$${zip_listing}"

ci: unit-test docs-check package-check

clean:
	rm -rf $(BUILD_DIR) release-assets
