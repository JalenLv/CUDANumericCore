#include "cncblas.cuh"
#include <iostream>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N (1 << 14)
#define MAX_NUMBER 5.0
#define MIN_NUMBER -0.5

int main() {
  float *x = new float[N];

  for (size_t i = 0; i < N; i++) {
    x[i] = -1.0;
  }
  x[5678] = MIN_NUMBER;
  x[1234] = MIN_NUMBER;

  float *d_x;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

  size_t result = cncblasSamin(N, d_x);

  std::cout << "The index of the minimum element is: " << result << std::endl;

  // use the official cublas library to verify the result
  cublasHandle_t handle;
  cublasCreate(&handle);
  int cublas_result;
  int min_index;
  cublas_result = cublasIsamin(handle, N, d_x, 1, &min_index);
  if (cublas_result != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasIsamax failed" << std::endl;
    return 1;
  }
  std::cout << "The index of the minimum element is: " << min_index - 1 << std::endl;
  cublasDestroy(handle);

  return 0;
}