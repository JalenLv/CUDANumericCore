project(cncblas CUDA)

set(src_files
        src/amaxmin.cu
        src/asum.cu
        src/axpy.cu
        src/copy.cu
        src/dot.cu
        src/norm2.cu
        src/rot.cu
        src/scal.cu
        src/swap.cu
        src/gemv.cu
        src/ger.cu
)

add_library(${PROJECT_NAME} SHARED ${src_files})

target_include_directories(${PROJECT_NAME} PRIVATE
        include
        ../../common
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# debugging
add_executable(cncblas_debug
        TEST.cu
)

target_include_directories(cncblas_debug PRIVATE
        ../../common
)
target_link_libraries(cncblas_debug ${PROJECT_NAME})

# Find and link the cuBLAS library
find_library(CUBLAS_LIBRARIES cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(cncblas_debug ${CUBLAS_LIBRARIES})

set_target_properties(cncblas_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
