#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;

TEST(scal, singlePrecision) {
  float *h_alpha, *h_x_cnc, *h_x_cublas;
  float *d_x_cnc, *d_x_cublas;

  h_x_cnc = new float[N];
  h_x_cublas = new float[N];
  h_alpha = new float;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRandf;
    h_x_cublas[i] = h_x_cnc[i];
  }
  *h_alpha = cncblasRandf;
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(float), cudaMemcpyHostToDevice));

  // Perform scal using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSscal(handle, N, h_alpha, d_x_cublas, 1);

  // Perform scal using cncblas
  cncblasSscal(N, h_alpha, d_x_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_x_cnc[i], h_x_cublas[i]);
  }

  delete[] h_x_cnc;
  delete[] h_x_cublas;
  delete h_alpha;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
}

TEST(scal, doublePrecision) {
  double *h_alpha, *h_x_cnc, *h_x_cublas;
  double *d_x_cnc, *d_x_cublas;

  h_x_cnc = new double[N];
  h_x_cublas = new double[N];
  h_alpha = new double;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRand;
    h_x_cublas[i] = h_x_cnc[i];
  }
  *h_alpha = cncblasRand;
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(double), cudaMemcpyHostToDevice));

  // Perform scal using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDscal(handle, N, h_alpha, d_x_cublas, 1);

  // Perform scal using cncblas
  cncblasDscal(N, h_alpha, d_x_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_DOUBLE_EQ(h_x_cnc[i], h_x_cublas[i]);
  }

  delete[] h_x_cnc;
  delete[] h_x_cublas;
  delete h_alpha;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
}

TEST(scal, complexSinglePrecision) {
  cuComplex *h_alpha, *h_x_cnc, *h_x_cublas;
  cuComplex *d_x_cnc, *d_x_cublas;

  h_x_cnc = new cuComplex[N];
  h_x_cublas = new cuComplex[N];
  h_alpha = new cuComplex;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_x_cublas[i] = h_x_cnc[i];
  }
  *h_alpha = make_cuComplex(cncblasRandf, cncblasRandf);
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Perform scal using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCscal(handle, N, h_alpha, d_x_cublas, 1);

  // Perform scal using cncblas
  cncblasCscal(N, h_alpha, d_x_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_x_cnc + i, h_x_cublas + i));
  }

  delete[] h_x_cnc;
  delete[] h_x_cublas;
  delete[] h_alpha;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
}

TEST(scal, complexDoublePrecision) {
  cuDoubleComplex *h_alpha, *h_x_cnc, *h_x_cublas;
  cuDoubleComplex *d_x_cnc, *d_x_cublas;

  h_x_cnc = new cuDoubleComplex[N];
  h_x_cublas = new cuDoubleComplex[N];
  h_alpha = new cuDoubleComplex;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_x_cublas[i] = h_x_cnc[i];
  }
  *h_alpha = make_cuDoubleComplex(cncblasRand, cncblasRand);
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Perform scal using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZscal(handle, N, h_alpha, d_x_cublas, 1);

  // Perform scal using cncblas
  cncblasZscal(N, h_alpha, d_x_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_x_cnc + i, h_x_cublas + i));
  }

  delete[] h_x_cnc;
  delete[] h_x_cublas;
  delete[] h_alpha;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
}

TEST(scal, complexSinglePrecisionReal) {
  float *h_alpha;
  cuComplex *h_x_cnc, *h_x_cublas;
  cuComplex *d_x_cnc, *d_x_cublas;

  h_x_cnc = new cuComplex[N];
  h_x_cublas = new cuComplex[N];
  h_alpha = new float;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_x_cublas[i] = h_x_cnc[i];
  }
  *h_alpha = cncblasRandf;
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Perform scal using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCsscal(handle, N, h_alpha, d_x_cublas, 1);

  // Perform scal using cncblas
  cncblasCsscal(N, h_alpha, d_x_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_x_cnc + i, h_x_cublas + i));
  }

  delete[] h_x_cnc;
  delete[] h_x_cublas;
  delete h_alpha;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
}

TEST(scal, complexDoublePrecisionReal) {
  double *h_alpha;
  cuDoubleComplex *h_x_cnc, *h_x_cublas;
  cuDoubleComplex *d_x_cnc, *d_x_cublas;

  h_x_cnc = new cuDoubleComplex[N];
  h_x_cublas = new cuDoubleComplex[N];
  h_alpha = new double;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_x_cublas[i] = h_x_cnc[i];
  }
  *h_alpha = cncblasRand;
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Perform scal using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZdscal(handle, N, h_alpha, d_x_cublas, 1);

  // Perform scal using cncblas
  cncblasZdscal(N, h_alpha, d_x_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_x_cnc + i, h_x_cublas + i));
  }

  delete[] h_x_cnc;
  delete[] h_x_cublas;
  delete h_alpha;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
}
