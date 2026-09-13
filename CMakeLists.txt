cmake_minimum_required(VERSION 3.18)

# Host-only configure must succeed without a CUDA toolkit. CUDA is an optional
# first-class language enabled only when CAPSNET_BUILD_CUDA=ON.
project(capsnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/CMakeModules")

include(GitInfo)
capsnet_get_git_info()

option(CAPSNET_BUILD_TESTING "Build host unit tests" ON)
option(CAPSNET_BUILD_CUDA "Enable CUDA language and kernel library" OFF)
option(CAPSNET_BUILD_CUDA_APP "Build historical NeuralNets executable" OFF)
option(CAPSNET_BUILD_GA "Build GA / NSGA-II host core" ON)
option(CAPSNET_BUILD_GA_DB "Build PostgreSQL / libpqxx fitness cache" OFF)
option(CAPSNET_BUILD_CUDA_TESTS "Build CUDA primitive parity tests" OFF)
option(CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR "Enable historical compatibility switches" OFF)
option(CAPSNET_ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(CAPSNET_ENABLE_ASAN "Build host targets with AddressSanitizer" OFF)

if(CAPSNET_BUILD_CUDA_APP AND NOT CAPSNET_BUILD_CUDA)
    message(FATAL_ERROR "CAPSNET_BUILD_CUDA_APP requires CAPSNET_BUILD_CUDA=ON")
endif()
if(CAPSNET_BUILD_CUDA_TESTS AND NOT CAPSNET_BUILD_CUDA)
    message(FATAL_ERROR "CAPSNET_BUILD_CUDA_TESTS requires CAPSNET_BUILD_CUDA=ON")
endif()
if(CAPSNET_BUILD_GA_DB AND NOT CAPSNET_BUILD_GA)
    message(FATAL_ERROR "CAPSNET_BUILD_GA_DB requires CAPSNET_BUILD_GA=ON")
endif()

file(READ "${CMAKE_SOURCE_DIR}/VERSION" CAPSNET_VERSION)
string(STRIP "${CAPSNET_VERSION}" CAPSNET_VERSION)

set(CAPSNET_CUDA_ENABLED_INT 0)
set(CAPSNET_CUDA_ARCHITECTURES_STRING "none")
if(CAPSNET_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CAPSNET_CUDA_ENABLED_INT 1)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        # Historical sm_30 is unsupported by modern NVCC. Default to a common
        # contemporary arch; override via -DCMAKE_CUDA_ARCHITECTURES=...
        set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures" FORCE)
    endif()
    set(CAPSNET_CUDA_ARCHITECTURES_STRING "${CMAKE_CUDA_ARCHITECTURES}")
endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/version.h.in"
    "${CMAKE_BINARY_DIR}/generated/capsnet_version.h"
    @ONLY
)

find_package(Armadillo REQUIRED)
find_package(Threads REQUIRED)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
if(CAPSNET_ENABLE_WARNINGS_AS_ERRORS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
if(CAPSNET_ENABLE_ASAN)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    add_link_options(-fsanitize=address)
endif()

# --- host core library -------------------------------------------------------
set(CAPSNET_HOST_CORE_SOURCES
    src/Utils.cpp
    src/CapsNetConfig.cpp
    src/MNISTDataIO.cpp
    src/MNISTReader.cpp
    src/models/Image.cpp
    src/models/PopulationStats.cpp
    src/CapsuleNetwork/BackpropUtils.cpp
    src/CapsuleNetwork/EvalPolicy.cpp
)

if(CAPSNET_BUILD_GA)
    list(APPEND CAPSNET_HOST_CORE_SOURCES
        src/GA/GACore.cpp
        src/GA/IndividualCore.cpp
        src/GA/PopulationCore.cpp
    )
endif()

add_library(capsnet_host_core STATIC ${CAPSNET_HOST_CORE_SOURCES})
target_include_directories(capsnet_host_core
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_BINARY_DIR}/generated"
)
target_link_libraries(capsnet_host_core
    PUBLIC
        ${ARMADILLO_LIBRARIES}
        Threads::Threads
)
if(ARMADILLO_INCLUDE_DIRS)
    target_include_directories(capsnet_host_core PUBLIC ${ARMADILLO_INCLUDE_DIRS})
endif()

if(CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR)
    target_compile_definitions(capsnet_host_core PUBLIC CAPSNET_PRESERVE_HISTORICAL_BEHAVIOR=1)
endif()

if(CAPSNET_BUILD_GA_DB)
    find_package(PQXX REQUIRED)
    target_sources(capsnet_host_core PRIVATE src/GA/CapsNetDAO.cpp)
    target_include_directories(capsnet_host_core PUBLIC ${PQXX_INCLUDE_DIRECTORIES})
    target_link_libraries(capsnet_host_core PUBLIC ${PQXX_LIBRARIES})
    target_compile_definitions(capsnet_host_core PUBLIC CAPSNET_HAS_GA_DB=1)
endif()

# --- optional CUDA kernel library -------------------------------------------
if(CAPSNET_BUILD_CUDA)
    add_library(capsnet_cuda_kernels STATIC
        src/CUDAUtils.cu
        src/models/CUUnifiedBlob.cu
        src/DeviceTimer.cu
    )
    target_include_directories(capsnet_cuda_kernels
        PUBLIC
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_BINARY_DIR}/generated"
    )
    target_link_libraries(capsnet_cuda_kernels PUBLIC capsnet_host_core)
    set_target_properties(capsnet_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# --- optional historical executable -----------------------------------------
if(CAPSNET_BUILD_CUDA_APP)
    set(CAPSNET_APP_SOURCES
        src/main.cu
        src/HostTimer.cpp
        src/ProgressBar.cpp
        src/ILayer.cpp
        src/models/FeatureMap.cpp
        src/models/Filter.cpp
        src/models/VectorMap.cpp
        src/CapsuleNetwork/Capsule.cpp
        src/CapsuleNetwork/CapsuleNetwork.cpp
        src/CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.cu
        src/ConvolutionalNetwork/ICNLayer.cpp
        src/ConvolutionalNetwork/ConvolutionalLayer.cpp
        src/ConvolutionalNetwork/ConvolutionalNetwork.cpp
        src/ConvolutionalNetwork/PoolingLayer.cpp
        src/ConvolutionalNetwork/CUConvolutionalNetwork/CUConvolutionalLayer.cu
        src/MultilayerPerceptron/Perceptron.cpp
        src/MultilayerPerceptron/PerceptronLayer.cpp
        src/MultilayerPerceptron/MultilayerPerceptron.cpp
        src/MultilayerPerceptron/FullyConnectedLayer.cpp
    )
    if(CAPSNET_BUILD_GA)
        list(APPEND CAPSNET_APP_SOURCES
            src/GA/GA.cu
            src/GA/Individual.cu
            src/GA/Population.cu
        )
    endif()
    add_executable(NeuralNets ${CAPSNET_APP_SOURCES})
    target_link_libraries(NeuralNets PRIVATE capsnet_cuda_kernels)
    set_target_properties(NeuralNets PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# --- tests -------------------------------------------------------------------
if(CAPSNET_BUILD_TESTING)
    include(CTest)
    enable_testing()

    add_library(doctest_headers INTERFACE)
    target_include_directories(doctest_headers INTERFACE "${CMAKE_SOURCE_DIR}/third_party/doctest")

    function(capsnet_add_host_test name)
        add_executable(${name} ${ARGN})
        target_link_libraries(${name} PRIVATE capsnet_host_core doctest_headers)
        target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/tests")
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        )
        add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(${name} PROPERTIES LABELS "host")
    endfunction()

    # Migrated doctest binary (discoverable cases when run as a single process).
    capsnet_add_host_test(test_utils tests/test_utils.cpp)

    # Remaining assert-based host binaries (stable CI coverage).
    capsnet_add_host_test(test_backprop tests/test_backprop.cpp)
    if(CAPSNET_BUILD_GA)
        capsnet_add_host_test(test_ga tests/test_ga.cpp)
        capsnet_add_host_test(test_ga_generation tests/test_ga_generation.cpp)
    endif()
    capsnet_add_host_test(test_mnist_io tests/test_mnist_io.cpp)
    capsnet_add_host_test(test_eval_no_mutation tests/test_eval_no_mutation.cpp)

    if(CAPSNET_BUILD_CUDA AND CAPSNET_BUILD_CUDA_TESTS)
        add_executable(test_cu_matrix_vector_mult tests/test_cu_matrix_vector_mult.cpp)
        target_link_libraries(test_cu_matrix_vector_mult PRIVATE capsnet_cuda_kernels doctest_headers)
        target_include_directories(test_cu_matrix_vector_mult PRIVATE "${CMAKE_SOURCE_DIR}/tests")
        set_target_properties(test_cu_matrix_vector_mult PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        )
        add_test(NAME test_cu_matrix_vector_mult COMMAND test_cu_matrix_vector_mult WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(test_cu_matrix_vector_mult PROPERTIES LABELS "cuda;gpu")
    endif()
endif()

message(STATUS "capsnet version: ${CAPSNET_VERSION}")
message(STATUS "git: ${CAPSNET_GIT_BRANCH} @ ${CAPSNET_GIT_COMMIT} dirty=${CAPSNET_GIT_DIRTY}")
message(STATUS "CUDA enabled: ${CAPSNET_BUILD_CUDA}")
message(STATUS "GA DB enabled: ${CAPSNET_BUILD_GA_DB}")
