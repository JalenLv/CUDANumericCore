#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;

TEST(swap, singlePrecision) {
  float *h_x, *h_y, *h_x_cublas, *h_y_cublas;
  float *d_x, *d_y, *d_x_cublas, *d_y_cublas;

  h_x = new float[N];
  h_y = new float[N];
  h_x_cublas = new float[N];
  h_y_cublas = new float[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRandf;
    h_y[i] = cncblasRandf;
    h_x_cublas[i] = h_x[i];
    h_y_cublas[i] = h_y[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(float), cudaMemcpyHostToDevice));

  // Perform swap using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSswap(handle, N, d_x_cublas, 1, d_y_cublas, 1);

  // Perform swap using cncblas
  cncblasSswap(N, d_x, d_y);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));

// Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_x[i], h_x_cublas[i]);
    EXPECT_FLOAT_EQ(h_y[i], h_y_cublas[i]);
  }

  delete[] h_x;
  delete[] h_y;
  delete[] h_x_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_x_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(swap, doublePrecision) {
  double *h_x, *h_y, *h_x_cublas, *h_y_cublas;
  double *d_x, *d_y, *d_x_cublas, *d_y_cublas;

  h_x = new double[N];
  h_y = new double[N];
  h_x_cublas = new double[N];
  h_y_cublas = new double[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRand;
    h_y[i] = cncblasRand;
    h_x_cublas[i] = h_x[i];
    h_y_cublas[i] = h_y[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(double), cudaMemcpyHostToDevice));

  // Perform swap using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDswap(handle, N, d_x_cublas, 1, d_y_cublas, 1);

  // Perform swap using cncblas
  cncblasDswap(N, d_x, d_y);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_DOUBLE_EQ(h_x[i], h_x_cublas[i]);
    EXPECT_DOUBLE_EQ(h_y[i], h_y_cublas[i]);
  }

  delete[] h_x;
  delete[] h_y;
  delete[] h_x_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_x_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(swap, complexSinglePrecision) {
  cuComplex *h_x, *h_y, *h_x_cublas, *h_y_cublas;
  cuComplex *d_x, *d_y, *d_x_cublas, *d_y_cublas;

  h_x = new cuComplex[N];
  h_y = new cuComplex[N];
  h_x_cublas = new cuComplex[N];
  h_y_cublas = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_x_cublas[i] = h_x[i];
    h_y_cublas[i] = h_y[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Perform swap using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCswap(handle, N, d_x_cublas, 1, d_y_cublas, 1);

  // Perform swap using cncblas
  cncblasCswap(N, d_x, d_y);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x, d_x, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y, d_y, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_x + i, h_x_cublas + i));
    EXPECT_TRUE(cncblasComplexIsEqual(h_y + i, h_y_cublas + i));
  }

  delete[] h_x;
  delete[] h_y;
  delete[] h_x_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_x_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(swap, complexDoublePrecision) {
  cuDoubleComplex *h_x, *h_y, *h_x_cublas, *h_y_cublas;
  cuDoubleComplex *d_x, *d_y, *d_x_cublas, *d_y_cublas;

  h_x = new cuDoubleComplex[N];
  h_y = new cuDoubleComplex[N];
  h_x_cublas = new cuDoubleComplex[N];
  h_y_cublas = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_x_cublas[i] = h_x[i];
    h_y_cublas[i] = h_y[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Perform swap using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZswap(handle, N, d_x_cublas, 1, d_y_cublas, 1);

  // Perform swap using cncblas
  cncblasZswap(N, d_x, d_y);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x, d_x, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y, d_y, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_x + i, h_x_cublas + i));
    EXPECT_TRUE(cncblasComplexIsEqual(h_y + i, h_y_cublas + i));
  }

  delete[] h_x;
  delete[] h_y;
  delete[] h_x_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_x_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}
