#include "cncblas.h"
#include <iostream>
#include <cstdio>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <string>

#define M (1 << 12)
#define N (1 << 12)

const float PI = 3.14159265358979323846;

int main() {
  // row major
  float *h_A_row, *h_x, *h_y;
  float *d_A_row, *d_x, *d_y;
  // column major
  float *h_A_col;
  float *d_A_col;
  // scalar
  float alpha = rand() / (float) RAND_MAX;

  // allocate memory
  h_A_row = new float[M * N];
  h_x = new float[M];
  h_y = new float[N];
  h_A_col = new float[M * N];
  cudaMalloc(&d_A_row, M * N * sizeof(float));
  cudaMalloc(&d_x, M * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_A_col, M * N * sizeof(float));

  // initialize data
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_row[row * N + col] = rand() / (float) RAND_MAX;
      h_A_col[col * M + row] = h_A_row[row * N + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x[i] = rand() / (float) RAND_MAX;
  }
  for (int i = 0; i < N; i++) {
    h_y[i] = rand() / (float) RAND_MAX;
  }

  // copy data to device
  cudaMemcpy(d_A_row, h_A_row, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_col, h_A_col, M * N * sizeof(float), cudaMemcpyHostToDevice);

  // compute ger on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSger(handle, M, N, &alpha, d_x, 1, d_y, 1, d_A_col, M);

  // compute ger on GPU using cncblas
  cncblasSger(M, N, &alpha, d_x, d_y, d_A_row);

  // copy data back to host
  cudaMemcpy(h_A_row, d_A_row, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_A_col, d_A_col, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // compare results
  float abs_err = 0;
  float max_err = 0;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      float diff = h_A_row[row * N + col] - h_A_col[col * M + row];
      abs_err += std::abs(diff);
      max_err = std::max(max_err, std::abs(diff));
    }
  }

  // print results
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Absolute error: " << abs_err << std::endl;
  std::cout << "Max error: " << max_err << std::endl;

  return 0;
}
