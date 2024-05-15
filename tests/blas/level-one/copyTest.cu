#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;

TEST(copy, singlePrecision) {
  float *h_x, *h_y_cnc, *h_y_cublas;
  float *d_x, *d_y_cnc, *d_y_cublas;

  h_x = new float[N];
  h_y_cnc = new float[N];
  h_y_cublas = new float[N];

  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRandf;
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

  // Compute copy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasScopy(handle, N, d_x, 1, d_y_cublas, 1);

  // Compute copy on GPU using cncblas
  cncblasScopy(N, d_x, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_y_cublas[i], h_y_cnc[i]);
  }

  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(copy, doublePrecision) {
  double *h_x, *h_y_cnc, *h_y_cublas;
  double *d_x, *d_y_cnc, *d_y_cublas;

  h_x = new double[N];
  h_y_cnc = new double[N];
  h_y_cublas = new double[N];

  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRand;
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));

  // Compute copy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDcopy(handle, N, d_x, 1, d_y_cublas, 1);

  // Compute copy on GPU using cncblas
  cncblasDcopy(N, d_x, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_DOUBLE_EQ(h_y_cublas[i], h_y_cnc[i]);
  }

  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(copy, complexSinglePrecision) {
  cuComplex *h_x, *h_y_cnc, *h_y_cublas;
  cuComplex *d_x, *d_y_cnc, *d_y_cublas;

  h_x = new cuComplex[N];
  h_y_cnc = new cuComplex[N];
  h_y_cublas = new cuComplex[N];

  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute copy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCcopy(handle, N, d_x, 1, d_y_cublas, 1);

  // Compute copy on GPU using cncblas
  cncblasCcopy(N, d_x, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cublas + i, h_y_cnc + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << "); but got: (" << h_y_cnc[i].x
                  << "," << h_y_cnc[i].y << ")" << std::endl << "Error: (" << std::abs(h_y_cublas[i].x - h_y_cnc[i].x)
                  << "," << std::abs(h_y_cublas[i].y - h_y_cnc[i].y) << ")" << std::endl;
  }

  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(copy, complexDoublePrecision) {
  cuDoubleComplex *h_x, *h_y_cnc, *h_y_cublas;
  cuDoubleComplex *d_x, *d_y_cnc, *d_y_cublas;

  h_x = new cuDoubleComplex[N];
  h_y_cnc = new cuDoubleComplex[N];
  h_y_cublas = new cuDoubleComplex[N];

  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute copy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZcopy(handle, N, d_x, 1, d_y_cublas, 1);

  // Compute copy on GPU using cncblas
  cncblasZcopy(N, d_x, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cublas + i, h_y_cnc + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << "); but got: (" << h_y_cnc[i].x
                  << "," << h_y_cnc[i].y << ")" << std::endl << "Error: (" << std::abs(h_y_cublas[i].x - h_y_cnc[i].x)
                  << "," << std::abs(h_y_cublas[i].y - h_y_cnc[i].y) << ")" << std::endl;
  }

  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}
