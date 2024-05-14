#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;

TEST(nrm2, singlePrecision) {
  float *h_x, *d_x;
  float *result_cnc, *result_cublas;

  h_x = new float[N];
  result_cublas = new float(0.0f);
  result_cnc = new float(1.0f);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRandf;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

  // Compute nrm2 on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSnrm2(handle, N, d_x, 1, result_cublas);

  // Compute nrm2 on GPU using cncblas
  *result_cnc = cncblasSnrm2(N, d_x);

  // Compare the results
  EXPECT_FLOAT_EQ(*result_cublas, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(nrm2, doublePrecision) {
  double *h_x, *d_x;
  double *result_cnc, *result_cublas;

  h_x = new double[N];
  result_cublas = new double(0.0);
  result_cnc = new double(1.0);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRand;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

  // Compute nrm2 on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDnrm2(handle, N, d_x, 1, result_cublas);

  // Compute nrm2 on GPU using cncblas
  *result_cnc = cncblasDnrm2(N, d_x);

  // Compare the results
  EXPECT_DOUBLE_EQ(*result_cublas, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(nrm2, complexSinglePrecision) {
  cuComplex *h_x, *d_x;
  float *result_cnc, *result_cublas;

  h_x = new cuComplex[N];
  result_cublas = new float(0.0f);
  result_cnc = new float(1.0f);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute nrm2 on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasScnrm2(handle, N, d_x, 1, result_cublas);

  // Compute nrm2 on GPU using cncblas
  *result_cnc = cncblasCnrm2(N, d_x);

  // Compare the results
  EXPECT_FLOAT_EQ(*result_cublas, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(nrm2, complexDoublePrecision) {
  cuDoubleComplex *h_x, *d_x;
  double *result_cnc, *result_cublas;

  h_x = new cuDoubleComplex[N];
  result_cublas = new double(0.0);
  result_cnc = new double(1.0);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute nrm2 on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDznrm2(handle, N, d_x, 1, result_cublas);

  // Compute nrm2 on GPU using cncblas
  *result_cnc = cncblasZnrm2(N, d_x);

  // Compare the results
  EXPECT_DOUBLE_EQ(*result_cublas, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}
