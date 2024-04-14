cmake_minimum_required(VERSION 3.28)
project(CUDANumericCore CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(CUDANumericCore SHARED library.cu)

set_target_properties(CUDANumericCore PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
