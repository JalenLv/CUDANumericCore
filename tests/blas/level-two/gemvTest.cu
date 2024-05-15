#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int M = 1 << 10;
const int N = 1 << 8;
const double epsilon = 1e-5;

TEST(gemv, singlePrecisionN) {
  float *alpha, *beta;
  float *h_x, *d_x;
  // row major - cncblas
  float *h_A_cnc, *h_y_cnc;
  float *d_A_cnc, *d_y_cnc;
  // col major - cublas
  float *h_A_cublas, *h_y_cublas;
  float *d_A_cublas, *d_y_cublas;

  alpha = new float;
  beta = new float;
  h_x = new float[N];
  h_A_cnc = new float[M * N];
  h_y_cnc = new float[M];
  h_A_cublas = new float[M * N];
  h_y_cublas = new float[M];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, M * sizeof(float)));

  srand(time(NULL));
  *alpha = cncblasRandf;
  *beta = cncblasRandf;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = cncblasRandf;
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRandf;
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = cncblasRandf;
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, M * sizeof(float), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemv(handle, CUBLAS_OP_N, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasSgemv(CNCBLAS_OP_N, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, M * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < M; i++) {
    EXPECT_NEAR(h_y_cublas[i], h_y_cnc[i], epsilon);
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, doublePrecisionN) {
  double *alpha, *beta;
  double *h_x, *d_x;
  // row major - cncblas
  double *h_A_cnc, *h_y_cnc;
  double *d_A_cnc, *d_y_cnc;
  // col major - cublas
  double *h_A_cublas, *h_y_cublas;
  double *d_A_cublas, *d_y_cublas;

  alpha = new double;
  beta = new double;
  h_x = new double[N];
  h_A_cnc = new double[M * N];
  h_y_cnc = new double[M];
  h_A_cublas = new double[M * N];
  h_y_cublas = new double[M];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, M * sizeof(double)));

  srand(time(NULL));
  *alpha = cncblasRand;
  *beta = cncblasRand;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = cncblasRand;
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x[i] = cncblasRand;
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = cncblasRand;
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, M * sizeof(double), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgemv(handle, CUBLAS_OP_N, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasDgemv(CNCBLAS_OP_N, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, M * sizeof(double), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < M; i++) {
    EXPECT_NEAR(h_y_cublas[i], h_y_cnc[i], epsilon);
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, complexSinglePrecisionN) {
  cuComplex *alpha, *beta;
  cuComplex *h_x, *d_x;
  // row major - cncblas
  cuComplex *h_A_cnc, *h_y_cnc;
  cuComplex *d_A_cnc, *d_y_cnc;
  // col major - cublas
  cuComplex *h_A_cublas, *h_y_cublas;
  cuComplex *d_A_cublas, *d_y_cublas;

  alpha = new cuComplex;
  beta = new cuComplex;
  h_x = new cuComplex[N];
  h_A_cnc = new cuComplex[M * N];
  h_y_cnc = new cuComplex[M];
  h_A_cublas = new cuComplex[M * N];
  h_y_cublas = new cuComplex[M];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, M * sizeof(cuComplex)));

  srand(time(NULL));
  *alpha = make_cuComplex(cncblasRandf, cncblasRandf);
  *beta = make_cuComplex(cncblasRandf, cncblasRandf);
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = make_cuComplex(cncblasRandf, cncblasRandf);
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, M * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCgemv(handle, CUBLAS_OP_N, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasCgemv(CNCBLAS_OP_N, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, M * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < M; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_cublas + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << ")\nGot: ("
                  << h_y_cnc[i].x << "," << h_y_cnc[i].y << ")\nError: (" << std::abs(h_y_cnc[i].x - h_y_cublas[i].x)
                  << "," << std::abs(h_y_cnc[i].y - h_y_cublas[i].y) << ")";
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, complexDoublePrecisionN) {
  cuDoubleComplex *alpha, *beta;
  cuDoubleComplex *h_x, *d_x;
  // row major - cncblas
  cuDoubleComplex *h_A_cnc, *h_y_cnc;
  cuDoubleComplex *d_A_cnc, *d_y_cnc;
  // col major - cublas
  cuDoubleComplex *h_A_cublas, *h_y_cublas;
  cuDoubleComplex *d_A_cublas, *d_y_cublas;

  alpha = new cuDoubleComplex;
  beta = new cuDoubleComplex;
  h_x = new cuDoubleComplex[N];
  h_A_cnc = new cuDoubleComplex[M * N];
  h_y_cnc = new cuDoubleComplex[M];
  h_A_cublas = new cuDoubleComplex[M * N];
  h_y_cublas = new cuDoubleComplex[M];
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, M * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  *alpha = make_cuDoubleComplex(cncblasRand, cncblasRand);
  *beta = make_cuDoubleComplex(cncblasRand, cncblasRand);
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = make_cuDoubleComplex(cncblasRand, cncblasRand);
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZgemv(handle, CUBLAS_OP_N, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasZgemv(CNCBLAS_OP_N, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < M; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_cublas + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << ")\nGot: ("
                  << h_y_cnc[i].x << "," << h_y_cnc[i].y << ")\nError: (" << std::abs(h_y_cnc[i].x - h_y_cublas[i].x)
                  << "," << std::abs(h_y_cnc[i].y - h_y_cublas[i].y) << ")";
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, singlePrecisionT) {
  float *alpha, *beta;
  float *h_x, *d_x;
  // row major - cncblas
  float *h_A_cnc, *h_y_cnc;
  float *d_A_cnc, *d_y_cnc;
  // col major - cublas
  float *h_A_cublas, *h_y_cublas;
  float *d_A_cublas, *d_y_cublas;

  alpha = new float;
  beta = new float;
  h_x = new float[M];
  h_A_cnc = new float[M * N];
  h_y_cnc = new float[N];
  h_A_cublas = new float[M * N];
  h_y_cublas = new float[N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(float)));

  srand(time(NULL));
  *alpha = cncblasRandf;
  *beta = cncblasRandf;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = cncblasRandf;
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x[i] = cncblasRandf;
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRandf;
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(float), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemv(handle, CUBLAS_OP_T, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasSgemv(CNCBLAS_OP_T, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(h_y_cublas[i], h_y_cnc[i], epsilon);
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, doublePrecisionT) {
  double *alpha, *beta;
  double *h_x, *d_x;
  // row major - cncblas
  double *h_A_cnc, *h_y_cnc;
  double *d_A_cnc, *d_y_cnc;
  // col major - cublas
  double *h_A_cublas, *h_y_cublas;
  double *d_A_cublas, *d_y_cublas;

  alpha = new double;
  beta = new double;
  h_x = new double[M];
  h_A_cnc = new double[M * N];
  h_y_cnc = new double[N];
  h_A_cublas = new double[M * N];
  h_y_cublas = new double[N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(double)));

  srand(time(NULL));
  *alpha = cncblasRand;
  *beta = cncblasRand;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = cncblasRand;
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x[i] = cncblasRand;
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRand;
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(double), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgemv(handle, CUBLAS_OP_T, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasDgemv(CNCBLAS_OP_T, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(h_y_cublas[i], h_y_cnc[i], epsilon);
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, complexSinglePrecisionT) {
  cuComplex *alpha, *beta;
  cuComplex *h_x, *d_x;
  // row major - cncblas
  cuComplex *h_A_cnc, *h_y_cnc;
  cuComplex *d_A_cnc, *d_y_cnc;
  // col major - cublas
  cuComplex *h_A_cublas, *h_y_cublas;
  cuComplex *d_A_cublas, *d_y_cublas;

  alpha = new cuComplex;
  beta = new cuComplex;
  h_x = new cuComplex[M];
  h_A_cnc = new cuComplex[M * N];
  h_y_cnc = new cuComplex[N];
  h_A_cublas = new cuComplex[M * N];
  h_y_cublas = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuComplex)));

  srand(time(NULL));
  *alpha = make_cuComplex(cncblasRandf, cncblasRandf);
  *beta = make_cuComplex(cncblasRandf, cncblasRandf);
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = make_cuComplex(cncblasRandf, cncblasRandf);
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCgemv(handle, CUBLAS_OP_T, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasCgemv(CNCBLAS_OP_T, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_cublas + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << ")\nGot: ("
                  << h_y_cnc[i].x << "," << h_y_cnc[i].y << ")\nError: (" << std::abs(h_y_cnc[i].x - h_y_cublas[i].x)
                  << "," << std::abs(h_y_cnc[i].y - h_y_cublas[i].y) << ")";
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, complexDoublePrecisionT) {
  cuDoubleComplex *alpha, *beta;
  cuDoubleComplex *h_x, *d_x;
  // row major - cncblas
  cuDoubleComplex *h_A_cnc, *h_y_cnc;
  cuDoubleComplex *d_A_cnc, *d_y_cnc;
  // col major - cublas
  cuDoubleComplex *h_A_cublas, *h_y_cublas;
  cuDoubleComplex *d_A_cublas, *d_y_cublas;

  alpha = new cuDoubleComplex;
  beta = new cuDoubleComplex;
  h_x = new cuDoubleComplex[M];
  h_A_cnc = new cuDoubleComplex[M * N];
  h_y_cnc = new cuDoubleComplex[N];
  h_A_cublas = new cuDoubleComplex[M * N];
  h_y_cublas = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  *alpha = make_cuDoubleComplex(cncblasRand, cncblasRand);
  *beta = make_cuDoubleComplex(cncblasRand, cncblasRand);
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = make_cuDoubleComplex(cncblasRand, cncblasRand);
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZgemv(handle, CUBLAS_OP_T, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasZgemv(CNCBLAS_OP_T, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_cublas + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << ")\nGot: ("
                  << h_y_cnc[i].x << "," << h_y_cnc[i].y << ")\nError: (" << std::abs(h_y_cnc[i].x - h_y_cublas[i].x)
                  << "," << std::abs(h_y_cnc[i].y - h_y_cublas[i].y) << ")";
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, complexSinglePrecisionC) {
  cuComplex *alpha, *beta;
  cuComplex *h_x, *d_x;
  // row major - cncblas
  cuComplex *h_A_cnc, *h_y_cnc;
  cuComplex *d_A_cnc, *d_y_cnc;
  // col major - cublas
  cuComplex *h_A_cublas, *h_y_cublas;
  cuComplex *d_A_cublas, *d_y_cublas;

  alpha = new cuComplex;
  beta = new cuComplex;
  h_x = new cuComplex[M];
  h_A_cnc = new cuComplex[M * N];
  h_y_cnc = new cuComplex[N];
  h_A_cublas = new cuComplex[M * N];
  h_y_cublas = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuComplex)));

  srand(time(NULL));
  *alpha = make_cuComplex(cncblasRandf, cncblasRandf);
  *beta = make_cuComplex(cncblasRandf, cncblasRandf);
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = make_cuComplex(cncblasRandf, cncblasRandf);
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = make_cuComplex(cncblasRandf, cncblasRandf);
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCgemv(handle, CUBLAS_OP_C, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasCgemv(CNCBLAS_OP_C, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_cublas + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << ")\nGot: ("
                  << h_y_cnc[i].x << "," << h_y_cnc[i].y << ")\nError: (" << std::abs(h_y_cnc[i].x - h_y_cublas[i].x)
                  << "," << std::abs(h_y_cnc[i].y - h_y_cublas[i].y) << ")";
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(gemv, complexDoublePrecisionC) {
  cuDoubleComplex *alpha, *beta;
  cuDoubleComplex *h_x, *d_x;
  // row major - cncblas
  cuDoubleComplex *h_A_cnc, *h_y_cnc;
  cuDoubleComplex *d_A_cnc, *d_y_cnc;
  // col major - cublas
  cuDoubleComplex *h_A_cublas, *h_y_cublas;
  cuDoubleComplex *d_A_cublas, *d_y_cublas;

  alpha = new cuDoubleComplex;
  beta = new cuDoubleComplex;
  h_x = new cuDoubleComplex[M];
  h_A_cnc = new cuDoubleComplex[M * N];
  h_y_cnc = new cuDoubleComplex[N];
  h_A_cublas = new cuDoubleComplex[M * N];
  h_y_cublas = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cnc, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(cuDoubleComplex)));

  srand(time(NULL));
  *alpha = make_cuDoubleComplex(cncblasRand, cncblasRand);
  *beta = make_cuDoubleComplex(cncblasRand, cncblasRand);
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cnc[row * N + col] = make_cuDoubleComplex(cncblasRand, cncblasRand);
      h_A_cublas[col * M + row] = h_A_cnc[row * N + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
    h_y_cublas[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Compute gemv on GPU using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZgemv(handle, CUBLAS_OP_C, M, N, alpha, d_A_cublas, M, d_x, 1, beta, d_y_cublas, 1);

  // Compute gemv on GPU using cncblas
  cncblasZgemv(CNCBLAS_OP_C, M, N, alpha, d_A_cnc, d_x, beta, d_y_cnc);

  // Copy the results back to host
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Compare the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_cublas + i))
                  << "Expected: (" << h_y_cublas[i].x << "," << h_y_cublas[i].y << ")\nGot: ("
                  << h_y_cnc[i].x << "," << h_y_cnc[i].y << ")\nError: (" << std::abs(h_y_cnc[i].x - h_y_cublas[i].x)
                  << "," << std::abs(h_y_cnc[i].y - h_y_cublas[i].y) << ")";
  }

  delete alpha;
  delete beta;
  delete[] h_x;
  delete[] h_A_cnc;
  delete[] h_y_cnc;
  delete[] h_A_cublas;
  delete[] h_y_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}
