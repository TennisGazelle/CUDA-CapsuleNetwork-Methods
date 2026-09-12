cmake_minimum_required(VERSION 2.6)
project(NeuralNets)

# set project standards and locations
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules")
#set(CXX11_FLAGS -std=gnu++11)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread --std=c++11 -O3 -g")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std c++11 -arch sm_30 -O2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpqxx -lpthread -Wall -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES") # to get rid of issues with strcpy

# find the dependencies
find_package(Armadillo REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
find_package(PQXX REQUIRED)

# set the includes and the src
set(INCLUDES
        ${PROJECT_SOURCE_DIR}/include
        ${ARMADILLO_INCLUDE_DIRS}
        )
include_directories(${INCLUDES})

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# finally the exec
cuda_add_executable(
        ${PROJECT_NAME}
        ${SOURCES}
        ${INCLUDES}
        )

target_link_libraries(NeuralNets ${ARMADILLO_LIBRARIES} ${PQXX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
