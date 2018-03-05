cmake_minimum_required(VERSION 2.6)
project(NeuralNets)

# set project standards and locations
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules")
set(CXX11_FLAGS -std=gnu++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -Wall -O0")

# find the dependencies
find_package(Armadillo REQUIRED)
find_package(Threads REQUIRED)

# set the includes and the src
set(INCLUDES
        ${PROJECT_SOURCE_DIR}/include
        ${ARMADILLO_INCLUDE_DIRS})
include_directories(${INCLUDES})

file(GLOB_RECURSE SOURCES "src/*.cpp")

# finally the exec
add_executable(NeuralNets ${SOURCES} ${INCLUDES})
target_link_libraries(NeuralNets ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
