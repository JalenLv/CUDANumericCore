#include <level_one.cuh>
#include <cuComplex.h>
#include <stdexcept>
#include <iostream>

/* -------------------- AXPY -------------------- */

const int BLOCK_SIZE = 256;

__global__ void cncblasSaxpyKernel(size_t n, float alpha, const float *x, float *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = alpha * x[idx] + y[idx];
  }
}

void cncblasSaxpy(size_t n, const float *alpha, const float *x, float *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasSaxpyKernel<<<num_blocks, BLOCK_SIZE>>>(n, *alpha, x, y);
}

__global__ void cncblasDaxpyKernel(size_t n, double alpha, const double *x, double *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = alpha * x[idx] + y[idx];
  }
}

void cncblasDaxpy(size_t n, const double *alpha, const double *x, double *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasDaxpyKernel<<<num_blocks, BLOCK_SIZE>>>(n, *alpha, x, y);
}

__global__ void cncblasCaxpyKernel(size_t n, cuComplex alpha, const cuComplex *x, cuComplex *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = cuCaddf(cuCmulf(alpha, x[idx]), y[idx]);
  }
}

void cncblasCaxpy(size_t n, const cuComplex *alpha, const cuComplex *x, cuComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasCaxpyKernel<<<num_blocks, BLOCK_SIZE>>>(n, *alpha, x, y);
}

__global__ void cncblasZaxpyKernel(size_t n, cuDoubleComplex alpha, const cuDoubleComplex *x, cuDoubleComplex *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = cuCadd(cuCmul(alpha, x[idx]), y[idx]);
  }
}

void cncblasZaxpy(size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, cuDoubleComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasZaxpyKernel<<<num_blocks, BLOCK_SIZE>>>(n, *alpha, x, y);
}
