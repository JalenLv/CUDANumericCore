#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int M = 1 << 10;
const int N = 1 << 8;
const double epsilon = 1e-5;

TEST(ger, singlePrecision) {
  float *h_alpha, *h_x, *h_y;
  float *d_x, *d_y;
  // row major - cncblas
  float *h_A_cncblas;
  float *d_A_cncblas;
  // column major - cublas
  float *h_A_cublas;
  float *d_A_cublas;

  h_alpha = new float;
  h_x = new float[M];
  h_y = new float[N];
  h_A_cncblas = new float[M * N];
  h_A_cublas = new float[M * N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_A_cncblas, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(float)));

  *h_alpha = cncblasRandf;
  for (int i = 0; i < M; i++) {
    h_x[i] = cncblasRandf;
  }
  for (int i = 0; i < N; i++) {
    h_y[i] = cncblasRandf;
  }
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cncblas[row * N + col] = cncblasRandf;
      h_A_cublas[col * M + row] = h_A_cncblas[row * N + col];
    }
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cncblas, h_A_cncblas, M * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(float), cudaMemcpyHostToDevice));

  // Computer ger using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSger(handle, M, N, h_alpha, d_x, 1, d_y, 1, d_A_cublas, M);

  // Compute ger using cncblas
  cncblasSger(M, N, h_alpha, d_x, d_y, d_A_cncblas);

  // Copy the result from device to host
  checkCudaErrors(cudaMemcpy(h_A_cublas, d_A_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_A_cncblas, d_A_cncblas, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Check the result
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      EXPECT_NEAR(h_A_cublas[col * M + row], h_A_cncblas[row * N + col], epsilon);
    }
  }

  // Free the memory
  delete h_alpha;
  delete h_x;
  delete h_y;
  delete h_A_cncblas;
  delete h_A_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_A_cncblas));
  checkCudaErrors(cudaFree(d_A_cublas));
}

TEST(ger, doublePrecision) {
  double *h_alpha, *h_x, *h_y;
  double *d_x, *d_y;
  // row major - cncblas
  double *h_A_cncblas;
  double *d_A_cncblas;
  // column major - cublas
  double *h_A_cublas;
  double *d_A_cublas;

  h_alpha = new double;
  h_x = new double[M];
  h_y = new double[N];
  h_A_cncblas = new double[M * N];
  h_A_cublas = new double[M * N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_A_cncblas, M * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(double)));

  *h_alpha = cncblasRand;
  for (int i = 0; i < M; i++) {
    h_x[i] = cncblasRand;
  }
  for (int i = 0; i < N; i++) {
    h_y[i] = cncblasRand;
  }
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cncblas[row * N + col] = cncblasRand;
      h_A_cublas[col * M + row] = h_A_cncblas[row * N + col];
    }
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cncblas, h_A_cncblas, M * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(double), cudaMemcpyHostToDevice));

  // Computer ger using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDger(handle, M, N, h_alpha, d_x, 1, d_y, 1, d_A_cublas, M);

  // Compute ger using cncblas
  cncblasDger(M, N, h_alpha, d_x, d_y, d_A_cncblas);

  // Copy the result from device to host
  checkCudaErrors(cudaMemcpy(h_A_cublas, d_A_cublas, M * N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_A_cncblas, d_A_cncblas, M * N * sizeof(double), cudaMemcpyDeviceToHost));

  // Check the result
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      EXPECT_NEAR(h_A_cublas[col * M + row], h_A_cncblas[row * N + col], epsilon);
    }
  }

  // Free the memory
  delete h_alpha;
  delete h_x;
  delete h_y;
  delete h_A_cncblas;
  delete h_A_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_A_cncblas));
  checkCudaErrors(cudaFree(d_A_cublas));
}

TEST(ger, singlePrecisionComplexU) {
  cuComplex *h_alpha, *h_x, *h_y;
  cuComplex *d_x, *d_y;
  // row major - cncblas
  cuComplex *h_A_cncblas;
  cuComplex *d_A_cncblas;
  // column major - cublas
  cuComplex *h_A_cublas;
  cuComplex *d_A_cublas;

  h_alpha = new cuComplex;
  h_x = new cuComplex[M];
  h_y = new cuComplex[N];
  h_A_cncblas = new cuComplex[M * N];
  h_A_cublas = new cuComplex[M * N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cncblas, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuComplex)));

  *h_alpha = make_cuComplex(cncblasRandf, cncblasRandf);
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  for (int i = 0; i < N; i++) {
    h_y[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cncblas[row * N + col] = make_cuComplex(cncblasRandf, cncblasRandf);
      h_A_cublas[col * M + row] = h_A_cncblas[row * N + col];
    }
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cncblas, h_A_cncblas, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Computer ger using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCgeru(handle, M, N, h_alpha, d_x, 1, d_y, 1, d_A_cublas, M);

  // Compute ger using cncblas
  cncblasCgeru(M, N, h_alpha, d_x, d_y, d_A_cncblas);

  // Copy the result from device to host
  checkCudaErrors(cudaMemcpy(h_A_cublas, d_A_cublas, M * N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_A_cncblas, d_A_cncblas, M * N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the result
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      EXPECT_TRUE(cncblasComplexIsEqual(h_A_cncblas + row * N + col, h_A_cublas + col * M + row));
    }
  }

  // Free the memory
  delete h_alpha;
  delete h_x;
  delete h_y;
  delete h_A_cncblas;
  delete h_A_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_A_cncblas));
  checkCudaErrors(cudaFree(d_A_cublas));
}

TEST(ger, doublePrecisionComplexU) {
  cuDoubleComplex *h_alpha, *h_x, *h_y;
  cuDoubleComplex *d_x, *d_y;
  // row major - cncblas
  cuDoubleComplex *h_A_cncblas;
  cuDoubleComplex *d_A_cncblas;
  // column major - cublas
  cuDoubleComplex *h_A_cublas;
  cuDoubleComplex *d_A_cublas;

  h_alpha = new cuDoubleComplex;
  h_x = new cuDoubleComplex[M];
  h_y = new cuDoubleComplex[N];
  h_A_cncblas = new cuDoubleComplex[M * N];
  h_A_cublas = new cuDoubleComplex[M * N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cncblas, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuDoubleComplex)));

  *h_alpha = make_cuDoubleComplex(cncblasRand, cncblasRand);
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  for (int i = 0; i < N; i++) {
    h_y[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cncblas[row * N + col] = make_cuDoubleComplex(cncblasRand, cncblasRand);
      h_A_cublas[col * M + row] = h_A_cncblas[row * N + col];
    }
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cncblas, h_A_cncblas, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Computer ger using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZgeru(handle, M, N, h_alpha, d_x, 1, d_y, 1, d_A_cublas, M);

  // Compute ger using cncblas
  cncblasZgeru(M, N, h_alpha, d_x, d_y, d_A_cncblas);

  // Copy the result from device to host
  checkCudaErrors(cudaMemcpy(h_A_cublas, d_A_cublas, M * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_A_cncblas, d_A_cncblas, M * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the result
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      EXPECT_TRUE(cncblasComplexIsEqual(h_A_cncblas + row * N + col, h_A_cublas + col * M + row));
    }
  }

  // Free the memory
  delete h_alpha;
  delete h_x;
  delete h_y;
  delete h_A_cncblas;
  delete h_A_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_A_cncblas));
  checkCudaErrors(cudaFree(d_A_cublas));
}

TEST(ger, singlePrecisionComplexC) {
  cuComplex *h_alpha, *h_x, *h_y;
  cuComplex *d_x, *d_y;
  // row major - cncblas
  cuComplex *h_A_cncblas;
  cuComplex *d_A_cncblas;
  // column major - cublas
  cuComplex *h_A_cublas;
  cuComplex *d_A_cublas;

  h_alpha = new cuComplex;
  h_x = new cuComplex[M];
  h_y = new cuComplex[N];
  h_A_cncblas = new cuComplex[M * N];
  h_A_cublas = new cuComplex[M * N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cncblas, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuComplex)));

  *h_alpha = make_cuComplex(cncblasRandf, cncblasRandf);
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  for (int i = 0; i < N; i++) {
    h_y[i] = make_cuComplex(cncblasRandf, cncblasRandf);
  }
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cncblas[row * N + col] = make_cuComplex(cncblasRandf, cncblasRandf);
      h_A_cublas[col * M + row] = h_A_cncblas[row * N + col];
    }
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cncblas, h_A_cncblas, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Computer ger using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasCgerc(handle, M, N, h_alpha, d_x, 1, d_y, 1, d_A_cublas, M);

  // Compute ger using cncblas
  cncblasCgerc(M, N, h_alpha, d_x, d_y, d_A_cncblas);

  // Copy the result from device to host
  checkCudaErrors(cudaMemcpy(h_A_cublas, d_A_cublas, M * N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_A_cncblas, d_A_cncblas, M * N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the result
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      EXPECT_TRUE(cncblasComplexIsEqual(h_A_cncblas + row * N + col, h_A_cublas + col * M + row));
    }
  }

  // Free the memory
  delete h_alpha;
  delete h_x;
  delete h_y;
  delete h_A_cncblas;
  delete h_A_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_A_cncblas));
  checkCudaErrors(cudaFree(d_A_cublas));
}

TEST(ger, doublePrecisionComplexC) {
  cuDoubleComplex *h_alpha, *h_x, *h_y;
  cuDoubleComplex *d_x, *d_y;
  // row major - cncblas
  cuDoubleComplex *h_A_cncblas;
  cuDoubleComplex *d_A_cncblas;
  // column major - cublas
  cuDoubleComplex *h_A_cublas;
  cuDoubleComplex *d_A_cublas;

  h_alpha = new cuDoubleComplex;
  h_x = new cuDoubleComplex[M];
  h_y = new cuDoubleComplex[N];
  h_A_cncblas = new cuDoubleComplex[M * N];
  h_A_cublas = new cuDoubleComplex[M * N];
  checkCudaErrors(cudaMalloc(&d_x, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cncblas, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_A_cublas, M * N * sizeof(cuDoubleComplex)));

  *h_alpha = make_cuDoubleComplex(cncblasRand, cncblasRand);
  for (int i = 0; i < M; i++) {
    h_x[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  for (int i = 0; i < N; i++) {
    h_y[i] = make_cuDoubleComplex(cncblasRand, cncblasRand);
  }
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_A_cncblas[row * N + col] = make_cuDoubleComplex(cncblasRand, cncblasRand);
      h_A_cublas[col * M + row] = h_A_cncblas[row * N + col];
    }
  }
  checkCudaErrors(cudaMemcpy(d_x, h_x, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cncblas, h_A_cncblas, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_cublas, h_A_cublas, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Computer ger using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasZgerc(handle, M, N, h_alpha, d_x, 1, d_y, 1, d_A_cublas, M);

  // Compute ger using cncblas
  cncblasZgerc(M, N, h_alpha, d_x, d_y, d_A_cncblas);

  // Copy the result from device to host
  checkCudaErrors(cudaMemcpy(h_A_cublas, d_A_cublas, M * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_A_cncblas, d_A_cncblas, M * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the result
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      EXPECT_TRUE(cncblasComplexIsEqual(h_A_cncblas + row * N + col, h_A_cublas + col * M + row));
    }
  }

  // Free the memory
  delete h_alpha;
  delete h_x;
  delete h_y;
  delete h_A_cncblas;
  delete h_A_cublas;
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_A_cncblas));
  checkCudaErrors(cudaFree(d_A_cublas));
}
