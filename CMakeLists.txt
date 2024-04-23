cmake_minimum_required(VERSION 3.28)
project(CUDANumericCore CUDA)

set(CMAKE_CUDA_STANDARD 17)

# add_library(CUDANumericCore SHARED ${CMAKE_CURRENT_SOURCE_DIR}/core/blas/***)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/include)

add_executable(
  CUDANumericCore ${CMAKE_CURRENT_SOURCE_DIR}/TEST.cu
                  ${CMAKE_CURRENT_SOURCE_DIR}/common/src/cnc_builtin_types.cu)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)

set_target_properties(CUDANumericCore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
