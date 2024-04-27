#ifndef CNCBLAS_HELPERS_CUH
#define CNCBLAS_HELPERS_CUH

/* ------------------------- AMAX/AMIN ------------------------- */

#include "cuComplex.h"

__device__ __always_inline float cncblasCmag(const cuComplex *x) {
  float a = x->x;
  float b = x->y;
  float mag = fabsf(a) + fabsf(b);
  return mag;
}

__device__ __always_inline double cncblasZmag(const cuDoubleComplex *x) {
  double a = x->x;
  double b = x->y;
  double mag = fabs(a) + fabs(b);
  return mag;
}

#endif // CNCBLAS_HELPERS_CUH