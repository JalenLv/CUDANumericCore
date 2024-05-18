cmake_minimum_required(VERSION 3.28)
project(CUDANumericCore LANGUAGES CUDA CXX)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()
set(CMAKE_CUDA_ARCHITECTURES native)
if (NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif ()
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()
if (NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)
if (CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif ()

# Build the core libraries
add_subdirectory(core)

# Build the tests executable
option(BUILD_TESTS "Build the tests" ON)
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(extern/googletest)
  add_subdirectory(tests)
endif ()
