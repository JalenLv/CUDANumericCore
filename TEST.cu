#include "cncblas.cuh"
#include <iostream>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>

#define N (1 << 22)

const float PI = 3.14159265358979323846;

int main() {
  // Test the SWAP function, and use cublas
  // to verify the results

  // Initialize the vectors
  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for (size_t i = 0; i < N; i++) {
    x[i] = i;
    y[i] = i + 1;
  }

  // Swap the vectors
  cncblasSswap(N, x, y);

  // Verify the results
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSswap(handle, N, x, 1, y, 1);
  cublasDestroy(handle);

  bool passed = true;
  for (size_t i = 0; i < N; i++) {
    if (x[i] != i || y[i] != i + 1) {
      passed = false;
      break;
    }
  }

  if (passed) {
    std::cout << "PASSED" << std::endl;
  } else {
    std::cout << "FAILED" << std::endl;
  }

  return 0;
}