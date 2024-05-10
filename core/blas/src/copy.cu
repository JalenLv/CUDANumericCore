#include "../cncblas.h"
#include <stdexcept>
#include <iostream>

/* -------------------- COPY -------------------- */

void cncblasScopy(size_t n, const float *x, float *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  checkCudaErrors(cudaMemcpy(y, x, n * sizeof(float), cudaMemcpyDeviceToDevice));
}

void cncblasDcopy(size_t n, const double *x, double *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  checkCudaErrors(cudaMemcpy(y, x, n * sizeof(double), cudaMemcpyDeviceToDevice));
}

void cncblasCcopy(size_t n, const cuComplex *x, cuComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  checkCudaErrors(cudaMemcpy(y, x, n * sizeof(cuComplex), cudaMemcpyDeviceToDevice));
}

void cncblasZcopy(size_t n, const cuDoubleComplex *x, cuDoubleComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  checkCudaErrors(cudaMemcpy(y, x, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
}
