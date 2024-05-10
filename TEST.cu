#include "cncblas.h"
#include <iostream>
#include <stdio.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>

//#define N (1 << 12)
#define N (1 << 0)

const float PI = 3.14159265358979323846;

int main() {
  // row major
  float *h_A_row, *h_x, *h_y_row;
  float *d_A_row, *d_x, *d_y_row;
  h_A_row = new float[N * N];
  h_x = new float[N];
  h_y_row = new float[N];
  checkCudaErrors(cudaMalloc(&d_A_row, N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_row, N * sizeof(float)));

  // col major
  float *h_A_col, *h_y_col;
  float *d_A_col, *d_y_col;
  h_A_col = new float[N * N];
  h_y_col = new float[N];
  checkCudaErrors(cudaMalloc(&d_A_col, N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_col, N * sizeof(float)));

  // Initialize A, x, and y
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      h_A_row[row * N + col] = rand() / (float) RAND_MAX;
      h_A_col[col * N + row] = h_A_row[row * N + col];
    }
    h_x[row] = rand() / (float) RAND_MAX;
    h_y_row[row] = rand() / (float) RAND_MAX;
    h_y_col[row] = h_y_row[row];
  }

  // Copy A, x, and y to device
  checkCudaErrors(cudaMemcpy(d_A_row, h_A_row, N * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_row, h_y_row, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_col, h_A_col, N * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_col, h_y_col, N * sizeof(float), cudaMemcpyHostToDevice));

  // Initialize alpha and beta
  float alpha = rand() / (float) RAND_MAX;
  float beta = rand() / (float) RAND_MAX;

  // Perform y = alpha * A * x + beta * y on CPU
  float *h_y_cpu = new float[N];
  for (int row = 0; row < N; row++) {
    h_y_cpu[row] = beta * h_y_row[row];
    for (int col = 0; col < N; col++) {
      h_y_cpu[row] += alpha * h_A_row[row * N + col] * h_x[col];
    }
  }

  // Perform y = alpha * A * x + beta * y using cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, d_A_col, N, d_x, 1, &beta, d_y_col, 1);

  // Perform y = alpha * A * x + beta * y using self-implemented cncblasSgemv
  cncblasSgemv(CNCBLAS_OP_N, N, N, &alpha, d_A_row, d_x, &beta, d_y_row);

  // Copy the result back to host
  checkCudaErrors(cudaMemcpy(h_y_row, d_y_row, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_col, d_y_col, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare the results
  float error = 0;
  for (int i = 0; i < N; i++) {
    error += std::abs(h_y_row[i] - h_y_col[i]);
  }
  std::cout << "Error: " << error << std::endl;

  // Free memory
  cublasDestroy(handle);
  delete[] h_y_cpu;
  delete[] h_A_row;
  delete[] h_x;
  delete[] h_y_row;
  checkCudaErrors(cudaFree(d_A_row));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_row));
  delete[] h_A_col;
  delete[] h_y_col;
  checkCudaErrors(cudaFree(d_A_col));
  checkCudaErrors(cudaFree(d_y_col));

  return 0;
}
