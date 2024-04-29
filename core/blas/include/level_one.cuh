#ifndef CNCBLAS_LEVEL_ONE_CUH
#define CNCBLAS_LEVEL_ONE_CUH

/*
 * Level 1 BLAS
 */

#include "cuComplex.h"

/* |-------------------------------------------------------| */
/* |  Note: The vector x (and y) must be a device pointer. | */
/* |-------------------------------------------------------| */

/*
 * AMAX: returns the index of the first element
 * of the vector x that has the largest value.
 */
size_t cncblasSamax(size_t n, const float *x);
size_t cncblasDamax(size_t n, const double *x);
size_t cncblasCamax(size_t n, const cuComplex *x);
size_t cncblasZamax(size_t n, const cuDoubleComplex *x);

/*
 * AMIN: returns the index of the first element
 * of the vector x that has the smallest value.
 */
size_t cncblasSamin(size_t n, const float *x);
size_t cncblasDamin(size_t n, const double *x);
size_t cncblasCamin(size_t n, const cuComplex *x);
size_t cncblasZamin(size_t n, const cuDoubleComplex *x);

/*
 * ASUM: computes the sum of the absolute values
 * of the elements of the vector x.
 */
float cncblasSasum(size_t n, const float *x);
double cncblasDasum(size_t n, const double *x);
float cncblasCasum(size_t n, const cuComplex *x);
double cncblasZasum(size_t n, const cuDoubleComplex *x);

/*
 * AXPY: multiplies the vector x by the scalar alpha
 * and adds it to the vector y, overwriting the y
 * with the result.
 */
void cncblasSaxpy(size_t n, const float *alpha, const float *x, float *y);
void cncblasDaxpy(size_t n, const double *alpha, const double *x, double *y);
void cncblasCaxpy(size_t n, const cuComplex *alpha, const cuComplex *x,
                  cuComplex *y);
void cncblasZaxpy(size_t n, const cuDoubleComplex *alpha,
                  const cuDoubleComplex *x, cuDoubleComplex *y);

/*
 * COPY: copies the vector x into the vector y.
 */
void cncblasScopy(size_t n, const float *x, float *y);
void cncblasDcopy(size_t n, const double *x, double *y);
void cncblasCcopy(size_t n, const cuComplex *x, cuComplex *y);
void cncblasZcopy(size_t n, const cuDoubleComplex *x, cuDoubleComplex *y);

/*
 * DOT: computes the dot product of the vectors x and y.
 */
float cncblasSdot(size_t n, const float *x, const float *y);
double cncblasDdot(size_t n, const double *x, const double *y);
cuComplex cncblasCdotu(size_t n, const cuComplex *x, const cuComplex *y);
cuComplex cncblasCdotc(size_t n, const cuComplex *x, const cuComplex *y);
cuDoubleComplex cncblasZdotu(size_t n, const cuDoubleComplex *x, const cuDoubleComplex *y);
cuDoubleComplex cncblasZdotc(size_t n, const cuDoubleComplex *x, const cuDoubleComplex *y);

/*
 * NRM2: computes the Euclidean norm of the vector x.
 */
float cncblasSnrm2(size_t n, const float *x);
double cncblasDnrm2(size_t n, const double *x);
float cncblasCnrm2(size_t n, const cuComplex *x);
double cncblasZnrm2(size_t n, const cuDoubleComplex *x);

/*
 * ROT: applies Givens rotation matrix (i.e., rotation in the x, y plane couter-clockwise
 * by angle defined by cos(alpha)=c and sin(alpha)=s) to vectors x and y.
 */
void cncblasSrot(size_t n, float *x, float *y, const float *alpha);
void cncblasDrot(size_t n, double *x, double *y, const double *alpha);
void cncblasSrot(size_t n, float *x, float *y, const float *c, const float *s);
void cncblasDrot(size_t n, double *x, double *y, const double *c, const double *s);
// TODO: cncblas complex rot
// void cncblasCrot(size_t n, cuComplex *x, cuComplex *y, const float *c, const cuComplex *s);
// void cncblasCsrot(size_t n, cuComplex *x, cuComplex *y, const float *c, const float *s);
// void cncblasZrot(size_t n, cuDoubleComplex *x, cuDoubleComplex *y, const double *c, const cuDoubleComplex *s);
// void cncblasZdrot(size_t n, cuDoubleComplex *x, cuDoubleComplex *y, const double *c, const double *s);

float *cncblasSrotg(float *a, float *b);
double *cncblasDrotg(double *a, double *b);
// TODO: cncblas complex rotg
// cuComplex *cncblasCrotg(cuComplex *a, cuComplex *b);
// cuDoubleComplex *cncblasZrotg(cuDoubleComplex *a, cuDoubleComplex *b);

// TODO: cncblas complex rotm
// void cncblasSrotm(size_t n, float *x, float *y, const float *param);
// void cncblasDrotm(size_t n, double *x, double *y, const double *param);

// TODO: cncblas complex rotmg
// void cncblasSrotmg(float *d1, float *d2, float *x1, const float *y1, float *param);
// void cncblasDrotmg(double *d1, double *d2, double *x1, const double *y1, double *param);

void cncblasSscal(size_t n, const float *alpha, float *x);
void cncblasDscal(size_t n, const double *alpha, double *x);
void cncblasCscal(size_t n, const cuComplex *alpha, cuComplex *x);
void cncblasCsscal(size_t n, const float *alpha, cuComplex *x);
void cncblasZscal(size_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x);
void cncblasZdscal(size_t n, const double *alpha, cuDoubleComplex *x);

void cncblasSswap(size_t n, float *x, float *y);
void cncblasDswap(size_t n, double *x, double *y);
void cncblasCswap(size_t n, cuComplex *x, cuComplex *y);
void cncblasZswap(size_t n, cuDoubleComplex *x, cuDoubleComplex *y);

#endif // CNCBLAS_LEVEL_ONE_CUH
