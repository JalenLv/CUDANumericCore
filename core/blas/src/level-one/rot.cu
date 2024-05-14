#include "cncblas.h"
#include <iostream>
#include <stdexcept>
#include "src/helpers.cuh"

/* -------------------- ROT --------------------- */

const int BLOCK_SIZE = 256;

void cncblasSrot(size_t n, float *x, float *y, const float *alpha) {
  try {
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }

  float *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  float *h_c, *h_s;
  h_c = new float(std::cos(*h_alpha));
  h_s = new float(std::sin(*h_alpha));

  cncblasSrot(n, x, y, h_c, h_s);

  delete h_alpha;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_alpha));
}

__global__ void cncblasSrot_kernel(size_t n, float *x, float *y, const float *c, const float *s) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float x_i = x[idx];
    float y_i = y[idx];
    x[idx] = *c * x_i + *s * y_i;
    y[idx] = -*s * x_i + *c * y_i;
  }
}

void cncblasSrot(size_t n, float *x, float *y, const float *c, const float *s) {
  // Check if there is any invalid parameter
  rotParamErrorCheck(n, x, y, c, s);
  // Preprocess the scalar parameters
  float *h_c, *d_c, *h_s, *d_s;
  cncblasScalarPointerPreprocess(c, h_c, d_c);
  cncblasScalarPointerPreprocess(s, h_s, d_s);

  // Launch the kernel
  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasSrot_kernel<<<num_blocks, BLOCK_SIZE>>>(n, x, y, d_c, d_s);

  // Free the memory
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_c));
  checkCudaErrors(cudaFree(d_s));
}

void cncblasDrot(size_t n, double *x, double *y, const double *alpha) {
  try {
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }

  double *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  double *h_c, *h_s;
  h_c = new double(std::cos(*h_alpha));
  h_s = new double(std::sin(*h_alpha));

  cncblasDrot(n, x, y, h_c, h_s);

  delete h_alpha;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_alpha));
}

__global__ void cncblasDrot_kernel(size_t n, double *x, double *y, const double *c, const double *s) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    double x_i = x[idx];
    double y_i = y[idx];
    x[idx] = *c * x_i + *s * y_i;
    y[idx] = -*s * x_i + *c * y_i;
  }
}

void cncblasDrot(size_t n, double *x, double *y, const double *c, const double *s) {
  // Check if there is any invalid parameter
  rotParamErrorCheck(n, x, y, c, s);
  // Preprocess the scalar parameters
  double *h_c, *d_c, *h_s, *d_s;
  cncblasScalarPointerPreprocess(c, h_c, d_c);
  cncblasScalarPointerPreprocess(s, h_s, d_s);

  // Launch the kernel
  size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cncblasDrot_kernel<<<num_blocks, BLOCK_SIZE>>>(n, x, y, d_c, d_s);
}

/* -------------------- ROTG --------------------- */

void cncblasSrotg(float *a, float *b, float *alpha) {
  try {
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }

  float *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  float *h_c, *h_s;
  h_c = new float;
  h_s = new float;

  cncblasSrotg(a, b, h_c, h_s);

  *h_alpha = std::atan2(*h_s, *h_c);
  rotgScalarPointerPostprocess(alpha, h_alpha);

  // Free the memory
  delete h_alpha;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_alpha));
}

void cncblasDrotg(double *a, double *b, double *alpha) {
  try {
    if (alpha == nullptr) {
      throw std::invalid_argument("alpha must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }

  double *h_alpha, *d_alpha;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);

  double *h_c, *h_s;
  h_c = new double;
  h_s = new double;

  cncblasDrotg(a, b, h_c, h_s);

  *h_alpha = std::atan2(*h_s, *h_c);
  rotgScalarPointerPostprocess(alpha, h_alpha);

  // Free the memory
  delete h_alpha;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_alpha));
}

void cncblasSrotg(float *a, float *b, float *c, float *s) {
  // Check if there is any invalid parameter
  rotgParamErrorCheck(a, b, c, s);
  // Preprocess the scalar parameters
  float *h_c, *d_c, *h_s, *d_s;
  float *h_a, *d_a, *h_b, *d_b;
  cncblasScalarPointerPreprocess(c, h_c, d_c);
  cncblasScalarPointerPreprocess(s, h_s, d_s);
  cncblasScalarPointerPreprocess(a, h_a, d_a);
  cncblasScalarPointerPreprocess(b, h_b, d_b);

  // Check if a and b are valid
  try {
    if (*h_a == 0 && *h_b == 0) {
      throw std::invalid_argument("a and b must not be both zero");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }

  // Compute the Givens rotation matrix
  using std::abs, std::sqrt;
  float sigma = (abs(*h_a) > abs(*h_b)) ? sgnf(*h_a) : sgnf(*h_b);
  float r = sigma * sqrt((*h_a) * (*h_a) + (*h_b) * (*h_b));
  if (r == 0) {
    *h_c = 1.0;
    *h_s = 0.0;
  } else {
    *h_c = *h_a / r;
    *h_s = *h_b / r;
  }
  float z = 0.0f;
  if (abs(*h_a) > abs(*h_b)) {
    z = *h_s;
  } else if ((abs(*h_b) >= abs(*h_a)) && (*h_c != 0.0)) {
    z = 1.0f / *h_c;
  } else if (*h_c == 0) {
    z = 1.0f;
  }
  *h_a = r;
  *h_b = z;

  // Postprocess the scalar parameters
  rotgScalarPointerPostprocess(a, h_a);
  rotgScalarPointerPostprocess(b, h_b);
  rotgScalarPointerPostprocess(c, h_c);
  rotgScalarPointerPostprocess(s, h_s);

  // Free the memory
  delete h_a;
  delete h_b;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));
  checkCudaErrors(cudaFree(d_s));
}

void cncblasDrotg(double *a, double *b, double *c, double *s) {
  // Check if there is any invalid parameter
  rotgParamErrorCheck(a, b, c, s);
  // Preprocess the scalar parameters
  double *h_c, *d_c, *h_s, *d_s;
  double *h_a, *d_a, *h_b, *d_b;
  cncblasScalarPointerPreprocess(c, h_c, d_c);
  cncblasScalarPointerPreprocess(s, h_s, d_s);
  cncblasScalarPointerPreprocess(a, h_a, d_a);
  cncblasScalarPointerPreprocess(b, h_b, d_b);

  // Check if a and b are valid
  try {
    if (*h_a == 0 && *h_b == 0) {
      throw std::invalid_argument("a and b must not be both zero");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }

  // Compute the Givens rotation matrix
  using std::abs, std::sqrt;
  double sigma = (abs(*h_a) > abs(*h_b)) ? sgn(*h_a) : sgn(*h_b);
  double r = sigma * sqrt((*h_a) * (*h_a) + (*h_b) * (*h_b));
  if (r == 0) {
    *h_c = 1.0;
    *h_s = 0.0;
  } else {
    *h_c = *h_a / r;
    *h_s = *h_b / r;
  }
  double z = 0.0;
  if (abs(*h_a) > abs(*h_b)) {
    z = *h_s;
  } else if ((abs(*h_b) >= abs(*h_a)) && (*h_c != 0.0)) {
    z = 1.0 / *h_c;
  } else if (*h_c == 0) {
    z = 1.0;
  }
  *h_a = r;
  *h_b = z;

  // Postprocess the scalar parameters
  rotgScalarPointerPostprocess(a, h_a);
  rotgScalarPointerPostprocess(b, h_b);
  rotgScalarPointerPostprocess(c, h_c);
  rotgScalarPointerPostprocess(s, h_s);

  // Free the memory
  delete h_a;
  delete h_b;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));
  checkCudaErrors(cudaFree(d_s));
}
