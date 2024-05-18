#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int M = 1 << 10;
const int N = 1 << 11;
const int kl = 100;
const int ku = 120;
const double epsilon = 1e-5;

TEST(gbmv, singlePrecisionN) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  float *alpha = new float(cncblasRandf);
  float *beta = new float(cncblasRandf);

  // cncblas - 0 based
  float *h_A_cnc, *h_x_cnc, *h_y_cnc;
  float *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new float[nColsA * nRowsA];
  h_x_cnc = new float[N];
  h_y_cnc = new float[M];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(float)));

  // using gemv to verify the correctness of the cncblas implementation
  float *h_A_gemv, *h_x_gemv, *h_y_gemv;
  float *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new float[M * N];
  h_x_gemv = new float[N];
  h_y_gemv = new float[M];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, M * sizeof(float)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(float));
  memset(h_A_gemv, 0, M * N * sizeof(float));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandf;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRandf;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = cncblasRandf;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, M * sizeof(float), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasSgbmv(CNCBLAS_OP_N, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasSgemv(CNCBLAS_OP_N, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, M * sizeof(float), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < M; i++) {
    EXPECT_NEAR(h_y_cnc[i], h_y_gemv[i], epsilon) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, singlePrecisionT) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  float *alpha = new float(cncblasRandf);
  float *beta = new float(cncblasRandf);

  // cncblas - 0 based
  float *h_A_cnc, *h_x_cnc, *h_y_cnc;
  float *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new float[nColsA * nRowsA];
  h_x_cnc = new float[M];
  h_y_cnc = new float[N];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, M * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(float)));

  // using gemv to verify the correctness of the cncblas implementation
  float *h_A_gemv, *h_x_gemv, *h_y_gemv;
  float *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new float[M * N];
  h_x_gemv = new float[M];
  h_y_gemv = new float[N];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, M * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, N * sizeof(float)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(float));
  memset(h_A_gemv, 0, M * N * sizeof(float));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandf;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x_cnc[i] = cncblasRandf;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRandf;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, M * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, M * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, N * sizeof(float), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasSgbmv(CNCBLAS_OP_T, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasSgemv(CNCBLAS_OP_T, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(h_y_cnc[i], h_y_gemv[i], epsilon) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, doublePrecisionN) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  double *alpha = new double(cncblasRand);
  double *beta = new double(cncblasRand);

  // cncblas - 0 based
  double *h_A_cnc, *h_x_cnc, *h_y_cnc;
  double *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new double[nColsA * nRowsA];
  h_x_cnc = new double[N];
  h_y_cnc = new double[M];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(double)));

  // using gemv to verify the correctness of the cncblas implementation
  double *h_A_gemv, *h_x_gemv, *h_y_gemv;
  double *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new double[M * N];
  h_x_gemv = new double[N];
  h_y_gemv = new double[M];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, M * sizeof(double)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(double));
  memset(h_A_gemv, 0, M * N * sizeof(double));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRand;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRand;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = cncblasRand;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, M * sizeof(double), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasDgbmv(CNCBLAS_OP_N, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasDgemv(CNCBLAS_OP_N, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, M * sizeof(double), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < M; i++) {
    EXPECT_NEAR(h_y_cnc[i], h_y_gemv[i], epsilon) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, doublePrecisionT) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  double *alpha = new double(cncblasRand);
  double *beta = new double(cncblasRand);

  // cncblas - 0 based
  double *h_A_cnc, *h_x_cnc, *h_y_cnc;
  double *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new double[nColsA * nRowsA];
  h_x_cnc = new double[M];
  h_y_cnc = new double[N];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, M * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(double)));

  // using gemv to verify the correctness of the cncblas implementation
  double *h_A_gemv, *h_x_gemv, *h_y_gemv;
  double *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new double[M * N];
  h_x_gemv = new double[M];
  h_y_gemv = new double[N];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, M * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, N * sizeof(double)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(double));
  memset(h_A_gemv, 0, M * N * sizeof(double));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRand;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x_cnc[i] = cncblasRand;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRand;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, N * sizeof(double), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasDgbmv(CNCBLAS_OP_T, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasDgemv(CNCBLAS_OP_T, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(h_y_cnc[i], h_y_gemv[i], epsilon) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, complexSinglePrecisionN) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  cuComplex *alpha = new cuComplex(cncblasRandC);
  cuComplex *beta = new cuComplex(cncblasRandC);

  // cncblas - 0 based
  cuComplex *h_A_cnc, *h_x_cnc, *h_y_cnc;
  cuComplex *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new cuComplex[nColsA * nRowsA];
  h_x_cnc = new cuComplex[N];
  h_y_cnc = new cuComplex[M];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(cuComplex)));

  // using gemv to verify the correctness of the cncblas implementation
  cuComplex *h_A_gemv, *h_x_gemv, *h_y_gemv;
  cuComplex *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new cuComplex[M * N];
  h_x_gemv = new cuComplex[N];
  h_y_gemv = new cuComplex[M];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, M * sizeof(cuComplex)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(cuComplex));
  memset(h_A_gemv, 0, M * N * sizeof(cuComplex));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandC;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRandC;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = cncblasRandC;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, M * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasCgbmv(CNCBLAS_OP_N, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasCgemv(CNCBLAS_OP_N, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, M * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < M; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_gemv + i)) << "at index " << i << "\n"
                                                                        << "Expected: " << h_y_gemv[i].x << " + " << h_y_gemv[i].y << "i\n"
                                                                        << " Got: " << h_y_cnc[i].x << " + " << h_y_cnc[i].y << "i\n"
                                                                        << "Error: " << h_y_cnc[i].x - h_y_gemv[i].x << " + " << h_y_cnc[i].y - h_y_gemv[i].y << "i";
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, complexSinglePrecisionT) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  cuComplex *alpha = new cuComplex(cncblasRandC);
  cuComplex *beta = new cuComplex(cncblasRandC);

  // cncblas - 0 based
  cuComplex *h_A_cnc, *h_x_cnc, *h_y_cnc;
  cuComplex *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new cuComplex[nColsA * nRowsA];
  h_x_cnc = new cuComplex[M];
  h_y_cnc = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuComplex)));

  // using gemv to verify the correctness of the cncblas implementation
  cuComplex *h_A_gemv, *h_x_gemv, *h_y_gemv;
  cuComplex *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new cuComplex[M * N];
  h_x_gemv = new cuComplex[M];
  h_y_gemv = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, N * sizeof(cuComplex)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(cuComplex));
  memset(h_A_gemv, 0, M * N * sizeof(cuComplex));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandC;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x_cnc[i] = cncblasRandC;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRandC;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasCgbmv(CNCBLAS_OP_T, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasCgemv(CNCBLAS_OP_T, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_gemv + i)) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, complexSinglePrecisionC) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  cuComplex *alpha = new cuComplex(cncblasRandC);
  cuComplex *beta = new cuComplex(cncblasRandC);

  // cncblas - 0 based
  cuComplex *h_A_cnc, *h_x_cnc, *h_y_cnc;
  cuComplex *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new cuComplex[nColsA * nRowsA];
  h_x_cnc = new cuComplex[M];
  h_y_cnc = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuComplex)));

  // using gemv to verify the correctness of the cncblas implementation
  cuComplex *h_A_gemv, *h_x_gemv, *h_y_gemv;
  cuComplex *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new cuComplex[M * N];
  h_x_gemv = new cuComplex[M];
  h_y_gemv = new cuComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, M * sizeof(cuComplex)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, N * sizeof(cuComplex)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(cuComplex));
  memset(h_A_gemv, 0, M * N * sizeof(cuComplex));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandC;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x_cnc[i] = cncblasRandC;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRandC;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, M * sizeof(cuComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, N * sizeof(cuComplex), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasCgbmv(CNCBLAS_OP_C, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasCgemv(CNCBLAS_OP_C, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, N * sizeof(cuComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_gemv + i)) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, complexDoublePrecisionN) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  cuDoubleComplex *alpha = new cuDoubleComplex(cncblasRandZ);
  cuDoubleComplex *beta = new cuDoubleComplex(cncblasRandZ);

  // cncblas - 0 based
  cuDoubleComplex *h_A_cnc, *h_x_cnc, *h_y_cnc;
  cuDoubleComplex *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new cuDoubleComplex[nColsA * nRowsA];
  h_x_cnc = new cuDoubleComplex[N];
  h_y_cnc = new cuDoubleComplex[M];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, M * sizeof(cuDoubleComplex)));

  // using gemv to verify the correctness of the cncblas implementation
  cuDoubleComplex *h_A_gemv, *h_x_gemv, *h_y_gemv;
  cuDoubleComplex *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new cuDoubleComplex[M * N];
  h_x_gemv = new cuDoubleComplex[N];
  h_y_gemv = new cuDoubleComplex[M];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, M * sizeof(cuDoubleComplex)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(cuDoubleComplex));
  memset(h_A_gemv, 0, M * N * sizeof(cuDoubleComplex));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandZ;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRandZ;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = cncblasRandZ;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasZgbmv(CNCBLAS_OP_N, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasZgemv(CNCBLAS_OP_N, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < M; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_gemv + i)) << "at index " << i << "\n"
                                                                  << "Expected: " << h_y_gemv[i].x << " + " << h_y_gemv[i].y << "i\n"
                                                                  << " Got: " << h_y_cnc[i].x << " + " << h_y_cnc[i].y << "i\n"
                                                                  << "Error: " << h_y_cnc[i].x - h_y_gemv[i].x << " + " << h_y_cnc[i].y - h_y_gemv[i].y << "i";
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, complexDoublePrecisionT) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  cuDoubleComplex *alpha = new cuDoubleComplex(cncblasRandZ);
  cuDoubleComplex *beta = new cuDoubleComplex(cncblasRandZ);

  // cncblas - 0 based
  cuDoubleComplex *h_A_cnc, *h_x_cnc, *h_y_cnc;
  cuDoubleComplex *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new cuDoubleComplex[nColsA * nRowsA];
  h_x_cnc = new cuDoubleComplex[M];
  h_y_cnc = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuDoubleComplex)));

  // using gemv to verify the correctness of the cncblas implementation
  cuDoubleComplex *h_A_gemv, *h_x_gemv, *h_y_gemv;
  cuDoubleComplex *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new cuDoubleComplex[M * N];
  h_x_gemv = new cuDoubleComplex[M];
  h_y_gemv = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, N * sizeof(cuDoubleComplex)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(cuDoubleComplex));
  memset(h_A_gemv, 0, M * N * sizeof(cuDoubleComplex));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandZ;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x_cnc[i] = cncblasRandZ;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRandZ;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasZgbmv(CNCBLAS_OP_T, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasZgemv(CNCBLAS_OP_T, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_gemv + i)) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}

TEST(gbmv, complexDoublePrecisionC) {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  cuDoubleComplex *alpha = new cuDoubleComplex(cncblasRandZ);
  cuDoubleComplex *beta = new cuDoubleComplex(cncblasRandZ);

  // cncblas - 0 based
  cuDoubleComplex *h_A_cnc, *h_x_cnc, *h_y_cnc;
  cuDoubleComplex *d_A_cnc, *d_x_cnc, *d_y_cnc;
  h_A_cnc = new cuDoubleComplex[nColsA * nRowsA];
  h_x_cnc = new cuDoubleComplex[M];
  h_y_cnc = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_cnc, nColsA * nRowsA * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_cnc, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(cuDoubleComplex)));

  // using gemv to verify the correctness of the cncblas implementation
  cuDoubleComplex *h_A_gemv, *h_x_gemv, *h_y_gemv;
  cuDoubleComplex *d_A_gemv, *d_x_gemv, *d_y_gemv;
  h_A_gemv = new cuDoubleComplex[M * N];
  h_x_gemv = new cuDoubleComplex[M];
  h_y_gemv = new cuDoubleComplex[N];
  checkCudaErrors(cudaMalloc(&d_A_gemv, M * N * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_x_gemv, M * sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMalloc(&d_y_gemv, N * sizeof(cuDoubleComplex)));

  memset(h_A_cnc, 0, nColsA * nRowsA * sizeof(cuDoubleComplex));
  memset(h_A_gemv, 0, M * N * sizeof(cuDoubleComplex));
  for (int col = 0; col < nColsA; col++) {
    for (int row = cncblasMax(0, col - ku); row <= cncblasMin(M - 1, col + kl); row++) {
      h_A_cnc[(row - col + ku) * nColsA + col] = cncblasRandZ;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < M; i++) {
    h_x_cnc[i] = cncblasRandZ;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < N; i++) {
    h_y_cnc[i] = cncblasRandZ;
    h_y_gemv[i] = h_y_cnc[i];
  }
  checkCudaErrors(cudaMemcpy(d_A_cnc, h_A_cnc, nColsA * nRowsA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_gemv, h_A_gemv, M * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_gemv, h_x_gemv, M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_gemv, h_y_gemv, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  // Perform gbmv using cncblas
  cncblasZgbmv(CNCBLAS_OP_C, M, N, kl, ku, alpha, d_A_cnc, d_x_cnc, beta, d_y_cnc);

  // Verify the results using gemv
  cncblasZgemv(CNCBLAS_OP_C, M, N, alpha, d_A_gemv, d_x_gemv, beta, d_y_gemv);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_gemv, d_y_gemv, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_TRUE(cncblasComplexIsEqual(h_y_cnc + i, h_y_gemv + i)) << "at index " << i;
  }

  // Free the memory
  delete[] h_A_cnc;
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_A_gemv;
  delete[] h_x_gemv;
  delete[] h_y_gemv;
  delete alpha;
  delete beta;
  checkCudaErrors(cudaFree(d_A_cnc));
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_A_gemv));
  checkCudaErrors(cudaFree(d_x_gemv));
  checkCudaErrors(cudaFree(d_y_gemv));
}
