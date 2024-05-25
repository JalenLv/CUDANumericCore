#include "cncblas.h"
#include <iostream>
#include <stdexcept>

/* -------------------- SWAP -------------------- */

const size_t BLOCK_SIZE = 256;

template<typename T>
__global__ void cncblasSwapKernel(size_t n, T *x, T *y) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    T temp = x[i];
    x[i] = y[i];
    y[i] = temp;
  }
}

template<typename T>
void cncblasSwap(size_t n, T *x, T *y) {
  // Check for invalid inputs
  try {
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasSwapKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, y);
}

// Explicit instantiations
template void cncblasSwap(size_t n, float *x, float *y);
template void cncblasSwap(size_t n, double *x, double *y);
template void cncblasSwap(size_t n, cuComplex *x, cuComplex *y);
template void cncblasSwap(size_t n, cuDoubleComplex *x, cuDoubleComplex *y);

void cncblasSswap(size_t n, float *x, float *y) {
  cncblasSwap<float>(n, x, y);
}

void cncblasDswap(size_t n, double *x, double *y) {
  cncblasSwap<double>(n, x, y);
}

void cncblasCswap(size_t n, cuComplex *x, cuComplex *y) {
  cncblasSwap<cuComplex>(n, x, y);
}

void cncblasZswap(size_t n, cuDoubleComplex *x, cuDoubleComplex *y) {
  cncblasSwap<cuDoubleComplex>(n, x, y);
}

