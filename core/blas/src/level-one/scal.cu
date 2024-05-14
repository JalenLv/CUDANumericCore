#include "cncblas.h"
#include <iostream>
#include <stdexcept>
#include "src/helpers.cuh"

/* -------------------- SCAL -------------------- */

const size_t BLOCK_SIZE = 256;

__global__ void cncblasSscalKernel(size_t n, const float *alpha, float *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= *alpha;
  }
}

void cncblasSscal(size_t n, const float *alpha, float *x) {
  // Check for invalid inputs
  scalParamErrorCheck(n, alpha, x);
  // Preprocess the scalar parameters
  float *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasSscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_alpha, x);
}

__global__ void cncblasDscalKernel(size_t n, const double *alpha, double *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= *alpha;
  }
}

void cncblasDscal(size_t n, const double *alpha, double *x) {
  // Check for invalid inputs
  scalParamErrorCheck(n, alpha, x);
  // Preprocess the scalar parameters
  double *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasDscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_alpha, x);
}

__global__ void cncblasCscalKernel(size_t n, const cuComplex *alpha, cuComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmulf(x[i], *alpha);
  }
}

void cncblasCscal(size_t n, const cuComplex *alpha, cuComplex *x) {
  // Check for invalid inputs
  scalParamErrorCheck(n, alpha, x);
  // Preprocess the scalar parameters
  cuComplex *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasCscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_alpha, x);
}

__global__ void cncblasCsscalKernel(size_t n, const float *alpha, cuComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmulf(x[i], make_cuComplex(*alpha, 0.0f));
  }
}

void cncblasCsscal(size_t n, const float *alpha, cuComplex *x) {
  // Check for invalid inputs
  scalParamErrorCheck(n, alpha, x);
  // Preprocess the scalar parameters
  float *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasCsscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_alpha, x);
}

__global__ void cncblasZscalKernel(size_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmul(x[i], *alpha);
  }
}

void cncblasZscal(size_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x) {
  // Check for invalid inputs
  scalParamErrorCheck(n, alpha, x);
  // Preprocess the scalar parameters
  cuDoubleComplex *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasZscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_alpha, x);
}

__global__ void cncblasZdscalKernel(size_t n, const double *alpha, cuDoubleComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmul(x[i], make_cuDoubleComplex(*alpha, 0.0));
  }
}

void cncblasZdscal(size_t n, const double *alpha, cuDoubleComplex *x) {
  // Check for invalid inputs
  scalParamErrorCheck(n, alpha, x);
  // Preprocess the scalar parameters
  double *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasZdscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_alpha, x);
}
