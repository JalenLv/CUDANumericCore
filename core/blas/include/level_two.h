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

/* |------------------------------------------------------| */
/* |  Note: The vectors x and y must be device pointers.  | */
/* |------------------------------------------------------| */

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
 * @param A The banded matrix A is stored column by column,
 *          with the main diagonal stored in row ku + 1 (starting in
 *          first position), the first superdiagonal stored in row ku
 *          (starting in second position), the first subdiagonal stored
 *          in row ku + 2 (starting in first position), etc. So that
 *          in general, the element A(i,j) is stored in the memory
 *          location A(ku+1+i-j,j) for j = 1,2,...,n and i
 *          in max(1,j-ku),...,min(m,j+kl). Also, the elements in the
 *          array A that do not conceptually correspond to the elements
 *          in the banded matrix (the top left ku x ku and bottom right
 *          kl x kl triangles) are not referenced.
 * @param x The vector x.
 * @param beta The scalar \f$ \beta \f$.
 * @param y The vector y.
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
 * @param m The number of rows of the matrix A.
 * @param n The number of columns of the matrix A.
 * @param alpha The scalar \f$ \alpha \f$.
 * @param A The matrix A.
 * @param x The vector x.
 * @param beta The scalar \f$ \beta \f$.
 * @param y The vector y.
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

/** @fn cncblas<T>ger
 * @brief This function performs the rank-1 update
 * \f$ A = \alpha \cdot x \cdot y^T + A \f$ if `ger()`, `geru()` is called
 * \f$ A = \alpha \cdot x \cdot y^H + A \f$ if `gerc()` is called
 *
 * @param m The number of rows of the matrix A.
 * @param n The number of columns of the matrix A.
 * @param alpha The scalar \f$ \alpha \f$.
 * @param x The vector x.
 * @param y The vector y.
 * @param A The matrix A.
 */
void cncblasSger(int m, int n,
                 const float *alpha, const float *x, const float *y,
                 float *A);
void cncblasDger(int m, int n,
                 const double *alpha, const double *x, const double *y,
                 double *A);
void cncblasCgeru(int m, int n,
                  const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                  cuComplex *A);
void cncblasCgerc(int m, int n,
                  const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                  cuComplex *A);
void cncblasZgeru(int m, int n,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                  cuDoubleComplex *A);
void cncblasZgerc(int m, int n,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                  cuDoubleComplex *A);

#endif