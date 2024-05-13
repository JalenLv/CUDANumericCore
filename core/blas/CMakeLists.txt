# Build the blas library
project(cncblas)
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
# Ensure the outside world cannot see the internal headers
target_include_directories(${PROJECT_NAME} PRIVATE
        include
        ../../common
)
# Let the outside world see the public headers
target_include_directories(${PROJECT_NAME} PUBLIC ./)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Instant debug
set(INSTANT_DEBUG cncblas_debug)
add_executable(${INSTANT_DEBUG} TEST.cu)
target_include_directories(${INSTANT_DEBUG} PRIVATE
        ../../common
)
target_link_libraries(${INSTANT_DEBUG} ${PROJECT_NAME})
# Find and link the cuBLAS library
find_library(CUBLAS_LIBRARIES cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(${INSTANT_DEBUG} ${CUBLAS_LIBRARIES})
set_target_properties(${INSTANT_DEBUG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
