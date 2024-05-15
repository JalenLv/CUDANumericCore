#include "cncblas.h"
#include <iostream>
#include <stdexcept>

/* ------------------------- NRM2 ------------------------- */

float cncblasSnrm2(size_t n, const float *x) {
  // check for invalid arguments
  try {
    if (x == nullptr) {
      throw std::invalid_argument("x must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
  return std::sqrt(cncblasSdot(n, x, x));
}

double cncblasDnrm2(size_t n, const double *x) {
  // check for invalid arguments
  try {
    if (x == nullptr) {
      throw std::invalid_argument("x must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
  return std::sqrt(cncblasDdot(n, x, x));
}

float cncblasCnrm2(size_t n, const cuComplex *x) {
  // check for invalid arguments
  try {
    if (x == nullptr) {
      throw std::invalid_argument("x must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
  return std::sqrt(cncblasCdotc(n, x, x).x);
}

double cncblasZnrm2(size_t n, const cuDoubleComplex *x) {
  // check for invalid arguments
  try {
    if (x == nullptr) {
      throw std::invalid_argument("x must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
  return std::sqrt(cncblasZdotc(n, x, x).x);
}