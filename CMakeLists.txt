cmake_minimum_required(VERSION 3.28)
project(CUDANumericCore LANGUAGES CXX)

set(CMAKE_CUDA_ARCHITECTURES 80 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_STANDARD 17)
enable_language(CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Build the core libraries
add_subdirectory(core)

# Build the test executable
add_executable(${PROJECT_NAME} TEST.cu)

# Include and link the core libraries
target_include_directories(${PROJECT_NAME} PUBLIC core/blas)
target_link_libraries(${PROJECT_NAME} cncblas)

# Find and link the cuBLAS library
find_library(CUBLAS_LIBRARIES cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${CUBLAS_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
