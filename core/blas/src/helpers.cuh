#ifndef CNCBLAS_HELPERS_CUH
#define CNCBLAS_HELPERS_CUH

/* ------------------------- AMAX/AMIN ------------------------- */

#include "cuComplex.h"

__device__ static __inline__ float cncblasCmag(const cuComplex *x) {
  float a = x->x;
  float b = x->y;
  float mag = fabsf(a) + fabsf(b);
  return mag;
}

__device__ static __inline__ double cncblasZmag(const cuDoubleComplex *x) {
  double a = x->x;
  double b = x->y;
  double mag = fabs(a) + fabs(b);
  return mag;
}

/* ------------------------- DOT ------------------------- */

__device__ static __inline__ void cncblasCVaddf(volatile cuComplex *a, volatile cuComplex *b) {
  a->x += b->x;
  a->y += b->y;
}

__device__ static __inline__ void cncblasZVadd(volatile cuDoubleComplex *a, volatile cuDoubleComplex *b) {
  a->x += b->x;
  a->y += b->y;
}

/* ------------------------- GEMV ------------------------- */

#include <iostream>
#include <stdexcept>

template<typename T>
inline static void
gemvParamErrorCheck(int m, int n,
                    const T *&alpha, const T *&A, const T *&x,
                    const T *&beta, T *&y) {
  try {
    if (m < 0 || n < 0) {
      throw std::invalid_argument("m or n is less than 0");
    }
    if (alpha == nullptr || beta == nullptr || A == nullptr || x == nullptr || y == nullptr) {
      throw std::invalid_argument("One or more input arrays are nullptr");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

template<typename T>
inline static void
gemvScalarPointerPreprocess(const T *&alpha, const T *beta,
                            T *&h_alpha, T *&h_beta, T *&d_alpha, T *&d_beta) {
  h_alpha = (T *) malloc(sizeof(T));
  h_beta = (T *) malloc(sizeof(T));
  checkCudaErrors(cudaMalloc(&d_alpha, sizeof(T)));
  checkCudaErrors(cudaMalloc(&d_beta, sizeof(T)));
  if (getMemoryType(alpha) == cudaMemoryTypeHost) {
    *h_alpha = *alpha;
    checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, sizeof(T), cudaMemcpyHostToDevice));
  } else if (getMemoryType(alpha) == cudaMemoryTypeDevice) {
    checkCudaErrors(cudaMemcpy(h_alpha, alpha, sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_alpha, alpha, sizeof(T), cudaMemcpyDeviceToDevice));
  } else if (getMemoryType(alpha) == cudaMemoryTypeUnregistered) {
    *h_alpha = *alpha;
    checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, sizeof(T), cudaMemcpyHostToDevice));
  }
  if (getMemoryType(beta) == cudaMemoryTypeHost) {
    *h_beta = *beta;
    checkCudaErrors(cudaMemcpy(d_beta, h_beta, sizeof(T), cudaMemcpyHostToDevice));
  } else if (getMemoryType(beta) == cudaMemoryTypeDevice) {
    checkCudaErrors(cudaMemcpy(h_beta, beta, sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_beta, beta, sizeof(T), cudaMemcpyDeviceToDevice));
  } else if (getMemoryType(beta) == cudaMemoryTypeUnregistered) {
    *h_beta = *beta;
    checkCudaErrors(cudaMemcpy(d_beta, h_beta, sizeof(T), cudaMemcpyHostToDevice));
  }
}

template<typename T>
inline static bool
gemvComplexIsEqual(const T *a, const T *b) {
  double epsilon = 1e-6;
  return (std::abs(a->x - b->x) < epsilon)
         && (std::abs(a->y - b->y) < epsilon);
}

#endif // CNCBLAS_HELPERS_CUH