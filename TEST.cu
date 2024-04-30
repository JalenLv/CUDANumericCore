#include "cncblas.cuh"
#include <iostream>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>

#define N (1 << 22)

const float PI = 3.14159265358979323846;

int main() {
  // Test the rotg function, and use cublas
  // to verify the results
  float a = 1.0f;
  float b = 1.0f;
  float c = 0.0f, s = 0.0f;
  cncblasSrotg(&a, &b, &c, &s);
  std::cout << "c: " << c << ", s: " << s << std::endl;
  std::cout << "r: " << a << ", z: " << b << std::endl;
  std::cout << "alpha: " << std::atan2(s, c) << std::endl;

  // Verify the results using cublas
  a = 1.0f, b = 1.0f;
  c = 0.0f, s = 0.0f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSrotg(handle, &a, &b, &c, &s);
  std::cout << "c: " << c << ", s: " << s << std::endl;
  std::cout << "r: " << a << ", z: " << b << std::endl;
  std::cout << "alpha: " << std::atan2(s, c) << std::endl;
  cublasDestroy(handle);

  return 0;
}