#include "cncblas.h"
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
  checkCudaErrors(cudaMallocManaged(&x, N * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&y, N * sizeof(float)));

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

  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, x);
  if (attributes.type == cudaMemoryTypeManaged) {
    std::cout << "x is managed memory" << std::endl;
  } else if (attributes.type == cudaMemoryTypeDevice) {
    std::cout << "x is device memory" << std::endl;
  } else if (attributes.type == cudaMemoryTypeHost) {
    std::cout << "x is host memory" << std::endl;
  }

  return 0;
}