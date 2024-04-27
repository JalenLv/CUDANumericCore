#include "cncblas.cuh"
#include <iostream>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N (1 << 22)

int main() {
  cuDoubleComplex *x = new cuDoubleComplex[N];
  cuDoubleComplex *d_x;
  for (size_t i = 0; i < N; i++) {
    x[i] = make_cuDoubleComplex(-i, i);
  }
  cudaMalloc(&d_x, N * sizeof(cuDoubleComplex));
  cudaMemcpy(d_x, x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  cuDoubleComplex *y = new cuDoubleComplex[N];
  cuDoubleComplex *d_y;
  for (size_t i = 0; i < N; i++) {
    y[i] = make_cuDoubleComplex(2 * i, 2 * i);
  }
  cudaMalloc(&d_y, N * sizeof(cuDoubleComplex));
  cudaMemcpy(d_y, y, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  cuDoubleComplex alpha = make_cuDoubleComplex(2.0, -2.0);
  cncblasZaxpy(N, &alpha, d_x, d_y);
  cudaMemcpy(y, d_y, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < 10; i++) {
    std::cout << y[i].x << ", " << y[i].y << std::endl;
  }

  for (size_t i = 0; i < N; i++) {
    y[i] = make_cuDoubleComplex(2 * i, 2 * i);
  }
  cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex));
  cudaMemcpy(d_y, y, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZaxpy(handle, N, &alpha, d_x, 1, d_y, 1);
  cudaMemcpy(y, d_y, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < 10; i++) {
    std::cout << y[i].x << ", " << y[i].y << std::endl;
  }

  return 0;
}