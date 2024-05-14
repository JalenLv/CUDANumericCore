#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;

TEST(axpy, singlePrecision) {
  float *alpha;
  float *h_x, *h_y_cnc, *h_y_cublas;
  float *d_x, *d_y_cnc, *d_y_cublas;

  alpha = new float(cncblasRandf);
  h_x = new float[N];
  h_y_cnc = new float[N];
  h_y_cublas = new float[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRandf;
    h_y_cnc[i] = cncblasRandf;
    h_y_cublas[i] = h_y_cnc[i];
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(float), cudaMemcpyHostToDevice));

  // Compute axpy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSaxpy(handle, N, alpha, d_x, 1, d_y_cublas, 1);

  // Compute axpy on GPU using cncblas
  cncblasSaxpy(N, alpha, d_x, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_y_cublas[i], h_y_cnc[i]);
  }

  delete alpha;
  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(axpy, doublePrecision) {
  double *alpha;
  double *h_x, *h_y_cnc, *h_y_cublas;
  double *d_x, *d_y_cnc, *d_y_cublas;

  alpha = new double(cncblasRand);
  h_x = new double[N];
  h_y_cnc = new double[N];
  h_y_cublas = new double[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRand;
    h_y_cnc[i] = cncblasRand;
    h_y_cublas[i] = h_y_cnc[i];
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(double), cudaMemcpyHostToDevice));

  // Compute axpy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDaxpy(handle, N, alpha, d_x, 1, d_y_cublas, 1);

  // Compute axpy on GPU using cncblas
  cncblasDaxpy(N, alpha, d_x, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_DOUBLE_EQ(h_y_cublas[i], h_y_cnc[i]);
  }

  delete alpha;
  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(axpy, complexSinglePrecision) {
  cuComplex *alpha;
  cuComplex *h_x, *h_y_cnc, *h_y_cublas;
  cuComplex *d_x, *d_y_cnc, *d_y_cublas;

  alpha = new cuComplex;
  *alpha = make_cuComplex(cncblasRandf, cncblasRandf);
  h_x = new cuComplex[N];
  h_y_cnc = new cuComplex[N];
  h_y_cublas = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y_cnc[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y_cublas[i] = h_y_cnc[i];
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute axpy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCaxpy(handle, N, alpha, d_x, 1, d_y_cublas, 1);

  // Compute axpy on GPU using cncblas
  cncblasCaxpy(N, alpha, d_x, d_y_cnc);

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

  delete alpha;
  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(axpy, complexDoublePrecision) {
  cuDoubleComplex *alpha;
  cuDoubleComplex *h_x, *h_y_cnc, *h_y_cublas;
  cuDoubleComplex *d_x, *d_y_cnc, *d_y_cublas;

  alpha = new cuDoubleComplex;
  *alpha = make_cuDoubleComplex(cncblasRand, cncblasRand);
  h_x = new cuDoubleComplex[N];
  h_y_cnc = new cuDoubleComplex[N];
  h_y_cublas = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y_cnc[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y_cublas[i] = h_y_cnc[i];
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute axpy on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZaxpy(handle, N, alpha, d_x, 1, d_y_cublas, 1);

  // Compute axpy on GPU using cncblas
  cncblasZaxpy(N, alpha, d_x, d_y_cnc);

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

  delete alpha;
  delete[] h_x;
  delete[] h_y_cnc;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_y_cublas));
}

