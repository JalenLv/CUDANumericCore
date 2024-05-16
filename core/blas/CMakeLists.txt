# Build the blas library
project(cncblas)
set(src_files
    src/level-one/amaxmin.cu
    src/level-one/asum.cu
    src/level-one/axpy.cu
    src/level-one/copy.cu
    src/level-one/dot.cu
    src/level-one/norm2.cu
    src/level-one/rot.cu
    src/level-one/scal.cu
    src/level-one/swap.cu
    src/level-two/gemv.cu
    src/level-two/ger.cu)
add_library(${PROJECT_NAME} SHARED ${src_files})
# Ensure the outside world cannot see the internal headers
target_include_directories(${PROJECT_NAME} PRIVATE include)
# Let the outside world see the public headers
target_include_directories(${PROJECT_NAME} PUBLIC ./)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Find the cuBLAS library
find_library(CUBLAS_LIBRARIES cublas
             PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# Instant debug
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(INSTANT_DEBUG cncblas_debug)
  add_executable(${INSTANT_DEBUG} TEST.cu)
  target_include_directories(${INSTANT_DEBUG} PRIVATE ../../common)
  target_link_libraries(${INSTANT_DEBUG} ${PROJECT_NAME})
  target_link_libraries(${INSTANT_DEBUG} ${CUBLAS_LIBRARIES})
  set_target_properties(${INSTANT_DEBUG} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
