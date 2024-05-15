#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;

TEST(dot, singlePrecision) {
  float *h_x, *h_y;
  float *d_x, *d_y;
  float *result_cnc, *result_cublas;

  h_x = new float[N];
  h_y = new float[N];
  result_cublas = new float(0.0f);
  result_cnc = new float(1.0f);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRandf;
    h_y[i] = cncblasRandf;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

  // Compute dot on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSdot(handle, N, d_x, 1, d_y, 1, result_cublas);

  // Compute dot on GPU using cncblas
  *result_cnc = cncblasSdot(N, d_x, d_y);

  // Compare the results
  EXPECT_NEAR(*result_cublas, *result_cnc, 1e-5);

  // Free memory
  delete[] h_x;
  delete[] h_y;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
}

TEST(dot, doublePrecision) {
  double *h_x, *h_y;
  double *d_x, *d_y;
  double *result_cnc, *result_cublas;

  h_x = new double[N];
  h_y = new double[N];
  result_cublas = new double(0.0);
  result_cnc = new double(1.0);
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRand;
    h_y[i] = cncblasRand;
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));

  // Compute dot on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDdot(handle, N, d_x, 1, d_y, 1, result_cublas);

  // Compute dot on GPU using cncblas
  *result_cnc = cncblasDdot(N, d_x, d_y);

  // Compare the results
  EXPECT_NEAR(*result_cublas, *result_cnc, 1e-10);

  // Free memory
  delete[] h_x;
  delete[] h_y;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
}

TEST(dot, complexSinglePrecisionU) {
  cuComplex *h_x, *h_y;
  cuComplex *d_x, *d_y;
  cuComplex *result_cnc, *result_cublas;

  h_x = new cuComplex[N];
  h_y = new cuComplex[N];
  result_cublas = new cuComplex(make_cuComplex(0.0f, 0.0f));
  result_cnc = new cuComplex(make_cuComplex(1.0f, 0.0f));
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute dot on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCdotu(handle, N, d_x, 1, d_y, 1, result_cublas);

  // Compute dot on GPU using cncblas
  *result_cnc = cncblasCdotu(N, d_x, d_y);

  // Compare the results
  EXPECT_TRUE(cncblasComplexIsEqual(result_cublas, result_cnc))
                << "Expected: (" << result_cublas->x << "," << result_cublas->y << "); but got: (" << result_cnc->x
                << "," << result_cnc->y << ")" << std::endl << "Error: (" << std::abs(result_cublas->x - result_cnc->x)
                << "," << std::abs(result_cublas->y - result_cnc->y) << ")" << std::endl;

  // Free memory
  delete[] h_x;
  delete[] h_y;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
}

TEST(dot, complexSinglePrecisionC) {
  cuComplex *h_x, *h_y;
  cuComplex *d_x, *d_y;
  cuComplex *result_cnc, *result_cublas;

  h_x = new cuComplex[N];
  h_y = new cuComplex[N];
  result_cublas = new cuComplex(make_cuComplex(0.0f, 0.0f));
  result_cnc = new cuComplex(make_cuComplex(1.0f, 0.0f));
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute dot on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCdotc(handle, N, d_x, 1, d_y, 1, result_cublas);

  // Compute dot on GPU using cncblas
  *result_cnc = cncblasCdotc(N, d_x, d_y);

  // Compare the results
  EXPECT_TRUE(cncblasComplexIsEqual(result_cublas, result_cnc))
                << "Expected: (" << result_cublas->x << "," << result_cublas->y << "); but got: (" << result_cnc->x
                << "," << result_cnc->y << ")" << std::endl << "Error: (" << std::abs(result_cublas->x - result_cnc->x)
                << "," << std::abs(result_cublas->y - result_cnc->y) << ")" << std::endl;

  // Free memory
  delete[] h_x;
  delete[] h_y;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
}

TEST(dot, complexDoublePrecisionU) {
  cuDoubleComplex *h_x, *h_y;
  cuDoubleComplex *d_x, *d_y;
  cuDoubleComplex *result_cnc, *result_cublas;

  h_x = new cuDoubleComplex[N];
  h_y = new cuDoubleComplex[N];
  result_cublas = new cuDoubleComplex(make_cuDoubleComplex(0.0, 0.0));
  result_cnc = new cuDoubleComplex(make_cuDoubleComplex(1.0, 0.0));
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute dot on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZdotu(handle, N, d_x, 1, d_y, 1, result_cublas);

  // Compute dot on GPU using cncblas
  *result_cnc = cncblasZdotu(N, d_x, d_y);

  // Compare the results
  EXPECT_TRUE(cncblasComplexIsEqual(result_cublas, result_cnc))
                << "Expected: (" << result_cublas->x << "," << result_cublas->y << "); but got: (" << result_cnc->x
                << "," << result_cnc->y << ")" << std::endl << "Error: (" << std::abs(result_cublas->x - result_cnc->x)
                << "," << std::abs(result_cublas->y - result_cnc->y) << ")" << std::endl;

  // Free memory
  delete[] h_x;
  delete[] h_y;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
}

TEST(dot, complexDoublePrecisionC) {
  cuDoubleComplex *h_x, *h_y;
  cuDoubleComplex *d_x, *d_y;
  cuDoubleComplex *result_cnc, *result_cublas;

  h_x = new cuDoubleComplex[N];
  h_y = new cuDoubleComplex[N];
  result_cublas = new cuDoubleComplex(make_cuDoubleComplex(0.0, 0.0));
  result_cnc = new cuDoubleComplex(make_cuDoubleComplex(1.0, 0.0));
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }

  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute dot on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZdotc(handle, N, d_x, 1, d_y, 1, result_cublas);

  // Compute dot on GPU using cncblas
  *result_cnc = cncblasZdotc(N, d_x, d_y);

  // Compare the results
  EXPECT_TRUE(cncblasComplexIsEqual(result_cublas, result_cnc))
                << "Expected: (" << result_cublas->x << "," << result_cublas->y << "); but got: (" << result_cnc->x
                << "," << result_cnc->y << ")" << std::endl << "Error: (" << std::abs(result_cublas->x - result_cnc->x)
                << "," << std::abs(result_cublas->y - result_cnc->y) << ")" << std::endl;

  // Free memory
  delete[] h_x;
  delete[] h_y;
  delete result_cublas;
  delete result_cnc;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
}
