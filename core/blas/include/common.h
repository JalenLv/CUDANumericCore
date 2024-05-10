#ifndef CNC_COMMON_H
#define CNC_COMMON_H

/* Prevent inclusion of internal headers from public headers. */
#ifndef CNCBLAS_INCLUDE_COMPILER_INTERNAL_HEADERS
#define CNCBLAS_INCLUDE_COMPILER_INTERNAL_HEADERS
#endif

/**
 * Error check macros:
 * This will output the proper CUDA error strings in
 * the event that a CUDA host call returns an error.
 */
#ifndef checkCudaErrors

#include <cuda.h>
#include <cstdio>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all the SDK helper functions
static inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (err != CUDA_SUCCESS) {
    const char *errorString = nullptr;
    cuGetErrorString(err, &errorString);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorString, file, line);
    exit(EXIT_FAILURE);
  }
}

static inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    const char *errorString;
    errorString = cudaGetErrorString(err);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorString, file, line);
    exit(EXIT_FAILURE);
  }
}

#endif

/**
 * Get the memory type to which the pointer points.
 */
#include <cuda_runtime.h>

__host__ static inline cudaMemoryType getMemoryType(const void *ptr) {
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, ptr);
  return attributes.type;
}

#endif //CNC_COMMON_H
