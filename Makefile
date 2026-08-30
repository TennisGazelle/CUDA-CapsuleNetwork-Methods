CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude
LDLIBS ?= -larmadillo

BUILD_DIR := .build
TEST_BIN := $(BUILD_DIR)/test_utils

.PHONY: help unit-test docs-check package package-check ci clean

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  unit-test     Build and run host-only characterization tests (no CUDA required)' \
	  '  docs-check    Validate required docs and relative Markdown links' \
	  '  package       Create versioned source tar.gz + zip under release-assets/' \
	  '  package-check Build packages and verify critical files are present' \
	  '  ci            Host PR validation bundle: unit-test + docs-check + package-check' \
	  '  clean         Remove revival-generated build/release outputs'

$(BUILD_DIR):
	mkdir -p $@

$(TEST_BIN): tests/test_utils.cpp src/Utils.cpp include/Utils.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_utils.cpp src/Utils.cpp -o $@ $(LDLIBS)

unit-test: $(TEST_BIN)
	$(TEST_BIN)

docs-check:
	python3 scripts/check_docs.py

package:
	bash scripts/package_release.sh

package-check: package | $(BUILD_DIR)
	@version="$$(tr -d '[:space:]' < VERSION)"; \
	 prefix="cuda-capsule-network-methods-$${version}"; \
	 test -s "release-assets/$${prefix}.tar.gz"; \
	 test -s "release-assets/$${prefix}.zip"; \
	 tar -tzf "release-assets/$${prefix}.tar.gz" > "$(BUILD_DIR)/tar-contents.txt"; \
	 unzip -l "release-assets/$${prefix}.zip" > "$(BUILD_DIR)/zip-contents.txt"; \
	 grep -q "$${prefix}/README.md" "$(BUILD_DIR)/tar-contents.txt"; \
	 grep -q "$${prefix}/SPEC.md" "$(BUILD_DIR)/zip-contents.txt"

ci: unit-test docs-check package-check

clean:
	rm -rf $(BUILD_DIR) release-assets
