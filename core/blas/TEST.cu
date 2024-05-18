#include "cncblas.h"
#include <iostream>
#include <cstdio>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <string>

const int M = 1 << 10;
const int N = 1 << 11;
const int kl = 100;
const int ku = 120;
//const int M = 4;
//const int N = 3;
//const int kl = 1;
//const int ku = 1;

const float PI = 3.14159265358979323846;

int main() {
  int nColsA = cncblasMin(N, M + ku);
  int nRowsA = ku + kl + 1;
  cuComplex *alpha = new cuComplex(cncblasRandC);
  cuComplex *beta = new cuComplex(cncblasRandC);
//  cuComplex one = make_cuComplex(1.0, 0.0);
//  cuComplex *alpha = new cuComplex(one);
//  cuComplex *beta = new cuComplex(one);


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
//      h_A_cnc[(row - col + ku) * nColsA + col] = one;
      h_A_gemv[row * N + col] = h_A_cnc[(row - col + ku) * nColsA + col];
    }
  }
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRandC;
//    h_x_cnc[i] = one;
    h_x_gemv[i] = h_x_cnc[i];
  }
  for (int i = 0; i < M; i++) {
    h_y_cnc[i] = cncblasRandC;
//    h_y_cnc[i] = one;
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
    if (!cncblasComplexIsEqual(h_y_cnc + i, h_y_gemv + i)) {
      std::cout << "Results do not match at " << i << std::endl;
    }
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
  return 0;
}
