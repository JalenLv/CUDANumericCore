#include "cncblas.cuh"
#include <iostream>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>

#define N (1 << 22)

int main() {
  cuComplex *cx, *cy;
  cuComplex *d_cx, *d_cy;

  cx = new cuComplex[N];
  cy = new cuComplex[N];

  for (int i = 0; i < N; i++) {
    cx[i].x = 1.0f;
    cx[i].y = 1.0f;
    cy[i].x = 2.0f;
    cy[i].y = 2.0f;
  }

  cudaMalloc(&d_cx, N * sizeof(cuComplex));
  cudaMalloc(&d_cy, N * sizeof(cuComplex));
  cudaMemcpy(d_cx, cx, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cy, cy, N * sizeof(cuComplex), cudaMemcpyHostToDevice);

  float result = 0.0f;
  result = cncblasCnrm2(N, d_cx);

  cublasHandle_t handle;
  cublasCreate(&handle);
  float result_cublas = 0.0f;
  cublasScnrm2(handle, N, d_cx, 1, &result_cublas);
  float epsilon = 1e-6;
  if (abs(result - result_cublas) < epsilon) {
    std::cout << "Test passed" << std::endl;
  } else {
    std::cout << "Test failed" << std::endl;
    std::cout << "cncblasCnrm2: " << std::fixed << std::setprecision(10) << result << std::endl;
    std::cout << "cublasScnrm2: " << std::fixed << std::setprecision(10) << result_cublas << std::endl;
    std::cout << "Difference: " << abs(result - result_cublas) << std::endl;
  }

  cuComplex cresult;
  cresult = cncblasCdotc(N, d_cx, d_cx);

  cuComplex cresult_cublas;
  cublasCdotc(handle, N, d_cx, 1, d_cx, 1, &cresult_cublas);
  if (abs(cresult.x - cresult_cublas.x) < epsilon && abs(cresult.y - cresult_cublas.y) < epsilon) {
    std::cout << "Test passed" << std::endl;
    std::cout << "cncblasCdotc: " << std::fixed << std::setprecision(10) << cresult.x << " + " << cresult.y << "i"
              << std::endl;
    std::cout << "cublasCdotc: " << std::fixed << std::setprecision(10) << cresult_cublas.x << " + " << cresult_cublas.y
              << "i" << std::endl;
  } else {
    std::cout << "Test failed" << std::endl;
    std::cout << "cncblasCdotu: " << cresult.x << " + " << cresult.y << "i" << std::endl;
    std::cout << "cublasCdotc: " << cresult_cublas.x << " + " << cresult_cublas.y << "i" << std::endl;
    std::cout << "Difference: " << abs(cresult.x - cresult_cublas.x) << " + " << abs(cresult.y - cresult_cublas.y)
              << "i" << std::endl;
  }

  return 0;
}