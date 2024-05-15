#ifndef CNCBLAS_COMMON_H
#define CNCBLAS_COMMON_H

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

__host__ static inline cudaMemoryType cncblasGetMemoryType(const void *ptr) {
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, ptr);
  return attributes.type;
}

/**
 * Preprocess the scalar pointer for the BLAS operations.
 */
template<typename T>
inline static void
cncblasScalarPointerPreprocess(const T *const &alpha,
                               T *&h_alpha, T *&d_alpha) {
  h_alpha = new T;
  checkCudaErrors(cudaMalloc(&d_alpha, sizeof(T)));
  if (cncblasGetMemoryType(alpha) == cudaMemoryTypeHost) {
    *h_alpha = *alpha;
    checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, sizeof(T), cudaMemcpyHostToDevice));
  } else if (cncblasGetMemoryType(alpha) == cudaMemoryTypeDevice) {
    checkCudaErrors(cudaMemcpy(h_alpha, alpha, sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_alpha, alpha, sizeof(T), cudaMemcpyDeviceToDevice));
  } else if (cncblasGetMemoryType(alpha) == cudaMemoryTypeUnregistered) {
    *h_alpha = *alpha;
    checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, sizeof(T), cudaMemcpyHostToDevice));
  } else if (cncblasGetMemoryType(alpha) == cudaMemoryTypeManaged) {
    *h_alpha = *alpha;
    checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, sizeof(T), cudaMemcpyHostToDevice));
  }
}

/**
 * Test if two complex numbers are equal.
 */
template<typename T>
inline static bool
cncblasComplexIsEqual(const T *a, const T *b) {
  double epsilon = 1e-8;
  T diff;
  diff.x = a->x - b->x;
  diff.y = a->y - b->y;
  return (diff.x * diff.x + diff.y * diff.y) < epsilon;
}

/**
 * Random number generator
 */
#ifndef cncblasRandf
#define cncblasRandf (rand() / (float) RAND_MAX - 0.5f)
#endif //cncblasRandf

#ifndef cncblasRand
#define cncblasRand (rand() / (double) RAND_MAX - 0.5)
#endif //cncblasRand

#endif //CNCBLAS_COMMON_H
