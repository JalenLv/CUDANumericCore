#include "cncblas.h"
#include "src/helpers.cuh"

/* -------------------- KERNEL DECLARATION -------------------- */

__global__ void cncblasSgerKernel(int m, int n,
                                  const float *alpha, const float *x, const float *y,
                                  float *A);
__global__ void cncblasDgerKernel(int m, int n,
                                  const double *alpha, const double *x, const double *y,
                                  double *A);
__global__ void cncblasCgeruKernel(int m, int n,
                                   const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                                   cuComplex *A);
__global__ void cncblasCgercKernel(int m, int n,
                                   const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                                   cuComplex *A);
__global__ void cncblasZgeruKernel(int m, int n,
                                   const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                                   cuDoubleComplex *A);
__global__ void cncblasZgercKernel(int m, int n,
                                   const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                                   cuDoubleComplex *A);

/* -------------------- GER -------------------- */

void cncblasSger(int m, int n,
                 const float *alpha, const float *x, const float *y,
                 float *A) {
  // check for invalid arguments
  gerParamErrorCheck(m, n, alpha, x, y, A);
  // allocate memory for scalar pointers
  float *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // quick return if possible
  if (m == 0 || n == 0 || *h_alpha == 0) {
    return;
  }

  // launch kernel
  dim3 BLOCK_SIZE(32, 16);
  dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                 (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
  cncblasSgerKernel<<<GRID_SIZE, BLOCK_SIZE>>>(m, n, d_alpha, x, y, A);
}

void cncblasDger(int m, int n,
                 const double *alpha, const double *x, const double *y,
                 double *A) {
  // check for invalid arguments
  gerParamErrorCheck(m, n, alpha, x, y, A);
  // allocate memory for scalar pointers
  double *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // quick return if possible
  if (m == 0 || n == 0 || *h_alpha == 0) {
    return;
  }

  // launch kernel
  dim3 BLOCK_SIZE(32, 16);
  dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                 (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
  cncblasDgerKernel<<<GRID_SIZE, BLOCK_SIZE>>>(m, n, d_alpha, x, y, A);
}

void cncblasCgeru(int m, int n,
                  const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                  cuComplex *A) {
  // check for invalid arguments
  gerParamErrorCheck(m, n, alpha, x, y, A);
  // allocate memory for scalar pointers
  cuComplex *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // quick return if possible
  cuComplex zero = make_cuComplex(0, 0);
  if (m == 0 || n == 0 || cncblasComplexIsEqual(h_alpha, &zero)) {
    return;
  }

  // launch kernel
  dim3 BLOCK_SIZE(32, 16);
  dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                 (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
  cncblasCgeruKernel<<<GRID_SIZE, BLOCK_SIZE>>>(m, n, d_alpha, x, y, A);
}

void cncblasCgerc(int m, int n,
                  const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                  cuComplex *A) {
  // check for invalid arguments
  gerParamErrorCheck(m, n, alpha, x, y, A);
  // allocate memory for scalar pointers
  cuComplex *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // quick return if possible
  cuComplex zero = make_cuComplex(0, 0);
  if (m == 0 || n == 0 || cncblasComplexIsEqual(h_alpha, &zero)) {
    return;
  }

  // launch kernel
  dim3 BLOCK_SIZE(32, 16);
  dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                 (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
  cncblasCgercKernel<<<GRID_SIZE, BLOCK_SIZE>>>(m, n, d_alpha, x, y, A);
}

void cncblasZgeru(int m, int n,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                  cuDoubleComplex *A) {
  // check for invalid arguments
  gerParamErrorCheck(m, n, alpha, x, y, A);
  // allocate memory for scalar pointers
  cuDoubleComplex *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // quick return if possible
  cuDoubleComplex zero = make_cuDoubleComplex(0, 0);
  if (m == 0 || n == 0 || cncblasComplexIsEqual(h_alpha, &zero)) {
    return;
  }

  // launch kernel
  dim3 BLOCK_SIZE(32, 16);
  dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                 (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
  cncblasZgeruKernel<<<GRID_SIZE, BLOCK_SIZE>>>(m, n, d_alpha, x, y, A);
}

void cncblasZgerc(int m, int n,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                  cuDoubleComplex *A) {
  // check for invalid arguments
  gerParamErrorCheck(m, n, alpha, x, y, A);
  // allocate memory for scalar pointers
  cuDoubleComplex *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // quick return if possible
  cuDoubleComplex zero = make_cuDoubleComplex(0, 0);
  if (m == 0 || n == 0 || cncblasComplexIsEqual(h_alpha, &zero)) {
    return;
  }

  // launch kernel
  dim3 BLOCK_SIZE(32, 16);
  dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                 (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
  cncblasZgercKernel<<<GRID_SIZE, BLOCK_SIZE>>>(m, n, d_alpha, x, y, A);
}

/* -------------------- KERNEL DEFINITION -------------------- */

__global__ void cncblasSgerKernel(int m, int n,
                                  const float *alpha, const float *x, const float *y,
                                  float *A) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    float a = A[row * n + col];
    float x_val = x[row];
    float y_val = y[col];
    A[row * n + col] = *alpha * x_val * y_val + a;
  }
}

__global__ void cncblasDgerKernel(int m, int n,
                                  const double *alpha, const double *x, const double *y,
                                  double *A) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    double a = A[row * n + col];
    double x_val = x[row];
    double y_val = y[col];
    A[row * n + col] = *alpha * x_val * y_val + a;
  }
}

__global__ void cncblasCgeruKernel(int m, int n,
                                   const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                                   cuComplex *A) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    cuComplex a = A[row * n + col];
    cuComplex x_val = x[row];
    cuComplex y_val = y[col];
    A[row * n + col] = cuCaddf(cuCmulf(*alpha, cuCmulf(x_val, y_val)), a);
  }
}

__global__ void cncblasCgercKernel(int m, int n,
                                   const cuComplex *alpha, const cuComplex *x, const cuComplex *y,
                                   cuComplex *A) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    cuComplex a = A[row * n + col];
    cuComplex x_val = x[row];
    cuComplex y_val = y[col];
    A[row * n + col] = cuCaddf(cuCmulf(*alpha, cuCmulf(x_val, cuConjf(y_val))), a);
  }
}

__global__ void cncblasZgeruKernel(int m, int n,
                                   const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                                   cuDoubleComplex *A) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    cuDoubleComplex a = A[row * n + col];
    cuDoubleComplex x_val = x[row];
    cuDoubleComplex y_val = y[col];
    A[row * n + col] = cuCadd(cuCmul(*alpha, cuCmul(x_val, y_val)), a);
  }
}

__global__ void cncblasZgercKernel(int m, int n,
                                   const cuDoubleComplex *alpha, const cuDoubleComplex *x, const cuDoubleComplex *y,
                                   cuDoubleComplex *A) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    cuDoubleComplex a = A[row * n + col];
    cuDoubleComplex x_val = x[row];
    cuDoubleComplex y_val = y[col];
    A[row * n + col] = cuCadd(cuCmul(*alpha, cuCmul(x_val, cuConj(y_val))), a);
  }
}

