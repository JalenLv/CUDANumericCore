#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;

TEST(amax, singlePrecision) {
  float *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new float[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRandf;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

  // Compute amax on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIsamax(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amax on GPU using cncblas
  *result_cnc = cncblasSamax(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(amax, doublePrecision) {
  double *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new double[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRand;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

  // Compute amax on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIdamax(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amax on GPU using cncblas
  *result_cnc = cncblasDamax(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(amax, complexSinglePrecision) {
  cuComplex *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new cuComplex[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute amax on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIcamax(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amax on GPU using cncblas
  *result_cnc = cncblasCamax(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(amax, complexDoublePrecision) {
  cuDoubleComplex *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new cuDoubleComplex[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute amax on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIzamax(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amax on GPU using cncblas
  *result_cnc = cncblasZamax(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}
