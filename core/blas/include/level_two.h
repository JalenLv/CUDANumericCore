#ifndef CNCBLAS_INCLUDE_COMPILER_INTERNAL_HEADERS
#ifdef _MSC_VER
#pragma message("blas/include/level_two.cuh is an internal header file and must not be used directly. Please use the public header file blas/cncblas.h instead.")
#else
#warning "blas/include/level_two.cuh is an internal header file and must not be used directly. Please use the public header file blas/cncblas.h instead."
#endif
#define CNCBLAS_INCLUDE_COMPILER_INTERNAL_HEADERS
#endif

#ifndef CNCBLAS_LEVEL_TWO_CUH
#define CNCBLAS_LEVEL_TWO_CUH

/*
 * Level 2 BLAS that perform matrix-vector operations.
 */

#include "cuComplex.h"

enum cncblasOperation_t {
  CNCBLAS_OP_N = 0,
  CNCBLAS_OP_T = 1,
  CNCBLAS_OP_C = 2
};

/* |-------------------------------------------------------| */
/* |  Note: The vectors x and y must be device pointers.  | */
/* |-------------------------------------------------------| */

/** @fn cncblas<T>gbmv
 * @brief This function performs the banded matrix-vector multiplication
 * \f$ y = \alpha \cdot \text{op}(A) \cdot x + \beta y \f$
 *
 * @param trans Specifies the operation to be performed as follows:
 *                - CNCBLAS_OP_N: \f$ \text{op}(A) = A \f$
 *                - CNCBLAS_OP_T: \f$ \text{op}(A) = A^T \f$
 *                - CNCBLAS_OP_C: \f$ \text{op}(A) = A^H \f$
 * @param m The number of rows of the matrix A.
 * @param n The number of columns of the matrix A.
 * @param kl The number of sub-diagonals of the matrix A.
 * @param ku The number of super-diagonals of the matrix A.
 * @param alpha The scalar \f$ \alpha \f$.
 * @param A
 **/
void cncblasSgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const float *alpha, const float *A, const float *x,
                  const float *beta, float *y);
void cncblasDgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const double *alpha, const double *A, const double *x,
                  const double *beta, double *y);
void cncblasCgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                  const cuComplex *beta, cuComplex *y);
void cncblasZgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                  const cuDoubleComplex *beta, cuDoubleComplex *y);
// TODO: Implement the function cncblas<T>gbmv

/** @fn cncblas<T>gemv
 * @brief This function performs the matrix-vector multiplication
 * \f$ y = \alpha \cdot \text{op}(A) \cdot x + \beta \cdot y \f$
 * where \f$ A \f$ is a general m*n matrix in row-major format, x
 * and y are vectors, and \f$ \alpha \f$ and \f$ \beta \f$ are
 * scales.
 *
 * @param trans Specifies the operation to be performed as follows:
 *                - CNCBLAS_OP_N: \f$ \text{op}(A) = A \f$
 *                - CNCBLAS_OP_T: \f$ \text{op}(A) = A^T \f$
 *                - CNCBLAS_OP_C: \f$ \text{op}(A) = A^H \f$
 **/
void cncblasSgemv(cncblasOperation_t trans,
                  int m, int n,
                  const float *alpha, const float *A, const float *x,
                  const float *beta, float *y);
void cncblasDgemv(cncblasOperation_t trans,
                  int m, int n,
                  const double *alpha, const double *A, const double *x,
                  const double *beta, double *y);
void cncblasCgemv(cncblasOperation_t trans,
                  int m, int n,
                  const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                  const cuComplex *beta, cuComplex *y);
void cncblasZgemv(cncblasOperation_t trans,
                  int m, int n,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                  const cuDoubleComplex *beta, cuDoubleComplex *y);

#endif