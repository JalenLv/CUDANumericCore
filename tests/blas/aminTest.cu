#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cncblas.h>

const int N = 1 << 10;

TEST(amin, singlePrecision) {
  float *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new float[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = rand() / (float) RAND_MAX;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

  // Compute amin on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIsamin(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amin on GPU using cncblas
  *result_cnc = cncblasSamin(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(amin, doublePrecision) {
  double *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new double[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = rand() / (double) RAND_MAX;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

  // Compute amin on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIdamin(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amin on GPU using cncblas
  *result_cnc = cncblasDamin(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc) << INFINITY;

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(amin, complexSinglePrecision) {
  cuComplex *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new cuComplex[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(rand() / (float) RAND_MAX, rand() / (float) RAND_MAX);
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute amin on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIcamin(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amin on GPU using cncblas
  *result_cnc = cncblasCamin(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}

TEST(amin, complexDoublePrecision) {
  cuDoubleComplex *h_x, *d_x;
  size_t *result_cnc, *result_cublas;

  h_x = new cuDoubleComplex[N];
  result_cublas = new size_t(0);
  result_cnc = new size_t(1);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(rand() / (double) RAND_MAX, rand() / (double) RAND_MAX);
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute amin on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasIzamin(handle, N, d_x, 1, reinterpret_cast<int *>(result_cublas));

  // Compute amin on GPU using cncblas
  *result_cnc = cncblasZamin(N, d_x);

  // Compare the results
  EXPECT_EQ(*result_cublas - 1, *result_cnc);

  // Free memory
  delete[] h_x;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
}
