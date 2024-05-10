#include "../cncblas.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

/* -------------------- ROT --------------------- */

const int BLOCK_SIZE = 256;

void cncblasSrot(size_t n, float *x, float *y, const float *alpha) {
  float c = std::cos(*alpha);
  float s = std::sin(*alpha);
  cncblasSrot(n, x, y, &c, &s);
}

__global__ void cncblasSrot_kernel(size_t n, float *x, float *y, float c, float s) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x_i = x[idx];
    float y_i = y[idx];
    x[idx] = c * x_i + s * y_i;
    y[idx] = -s * x_i + c * y_i;
  }
}

void cncblasSrot(size_t n, float *x, float *y, const float *c, const float *s) {
  // Check if there is any invalid parameter
  try {
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
      throw std::invalid_argument("x, y, c, and s must not be null");
    }
    const float epsilon = 1e-5;
    if (std::abs((*c) * (*c) + (*s) * (*s) - 1) > epsilon) {
      throw std::invalid_argument("c^2 + s^2 must be equal to 1");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return;
  }

  // Launch the kernel
  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasSrot_kernel<<<num_blocks, BLOCK_SIZE>>>(n, x, y, *c, *s);
}

void cncblasDrot(size_t n, double *x, double *y, const double *alpha) {
  double c = std::cos(*alpha);
  double s = std::sin(*alpha);
  cncblasDrot(n, x, y, &c, &s);
}

__global__ void cncblasDrot_kernel(size_t n, double *x, double *y, double c, double s) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double x_i = x[idx];
    double y_i = y[idx];
    x[idx] = c * x_i + s * y_i;
    y[idx] = -s * x_i + c * y_i;
  }
}

void cncblasDrot(size_t n, double *x, double *y, const double *c, const double *s) {
  // Check if there is any invalid parameter
  try {
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
      throw std::invalid_argument("x, y, c, and s must not be null");
    }
    const double epsilon = 1e-5;
    if (std::abs((*c) * (*c) + (*s) * (*s) - 1) > epsilon) {
      throw std::invalid_argument("c^2 + s^2 must be equal to 1");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return;
  }

  // Launch the kernel
  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasDrot_kernel<<<num_blocks, BLOCK_SIZE>>>(n, x, y, *c, *s);
}

/* -------------------- ROTG --------------------- */

#ifndef sgnf
#define sgnf(x) ((x) > 0.0f ? 1.0f : ((x) < 0.0f ? -1.0f : 0.0f))
#endif // sgnf(x)

#ifndef sgn
#define sgn(x) ((x) > 0.0 ? 1.0 : ((x) < 0.0 ? -1.0 : 0.0))
#endif // sgn(x)

void cncblasSrotg(float *a, float *b, float *alpha) {
  float c, s;
  cncblasSrotg(a, b, &c, &s);
  *alpha = std::atan2(s, c);
}

void cncblasDrotg(double *a, double *b, double *alpha) {
  double c, s;
  cncblasDrotg(a, b, &c, &s);
  *alpha = std::atan2(s, c);
}

void cncblasSrotg(float *a, float *b, float *c, float *s) {
  // Check if there is any invalid parameter
  try {
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
      throw std::invalid_argument("a, b, c, and s must not be null");
    }
    if (*a == 0 && *b == 0) {
      throw std::invalid_argument("a and b must not be both zero");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return;
  }

  // Compute the Givens rotation matrix
  float sigma = (std::abs(*a) > std::abs(*b)) ? sgnf(*a) : sgnf(*b);
  float r = sigma * std::sqrt((*a) * (*a) + (*b) * (*b));
  *c = *a / r;
  *s = *b / r;

  float z = 0.0f;
  if (std::abs(*a) > std::abs(*b)) {
    z = *s;
  } else if ((std::abs(*b) >= std::abs(*a)) && (*c != 0.0f)) {
    z = 1.0f / *c;
  } else if (*c == 0) {
    z = 1.0f;
  }

  // Update the values of a and b
  *a = r;
  *b = z;
}

void cncblasDrotg(double *a, double *b, double *c, double *s) {
  // Check if there is any invalid parameter
  try {
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
      throw std::invalid_argument("a, b, c, and s must not be null");
    }
    if (*a == 0 && *b == 0) {
      throw std::invalid_argument("a and b must not be both zero");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return;
  }

  // Compute the Givens rotation matrix
  double sigma = (*a > *b) ? sgn(*a) : sgn(*b);
  double r = sigma * std::sqrt((*a) * (*a) + (*b) * (*b));
  *c = *a / r;
  *s = *b / r;

  double z = 0.0;
  if (*a > *b) {
    z = *s;
  } else if ((*b >= *a) && (*c != 0.0)) {
    z = 1.0 / *c;
  } else if (*c == 0) {
    z = 1.0;
  }

  // Update the values of a and b
  *a = r;
  *b = z;
}
