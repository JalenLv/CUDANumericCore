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

#endif // CNCBLAS_HELPERS_CUH