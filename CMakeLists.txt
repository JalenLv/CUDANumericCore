cmake_minimum_required(VERSION 3.28)
project(CUDANumericCore LANGUAGES CUDA CXX)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()
set(CMAKE_CUDA_ARCHITECTURES 80 89)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Build the core libraries
add_subdirectory(core)

# Build the tests executable
option(BUILD_TESTS "Build the tests" ON)
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(extern/googletest)
  add_subdirectory(tests)
endif ()
