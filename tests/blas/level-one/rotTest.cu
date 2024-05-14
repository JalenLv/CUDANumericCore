#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cncblas.h"

const int N = 1 << 12;
const double PI = 3.14159265358979323846;

/* -------------------- ROT --------------------- */

TEST(rot, singlePrecision) {
  float *h_alpha, *h_c, *h_s;
  float *h_x_cnc, *h_y_cnc, *h_x_cublas, *h_y_cublas;
  float *d_x_cnc, *d_y_cnc, *d_x_cublas, *d_y_cublas;

  h_x_cnc = new float[N];
  h_y_cnc = new float[N];
  h_x_cublas = new float[N];
  h_y_cublas = new float[N];
  h_alpha = new float;
  h_c = new float;
  h_s = new float;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(float)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRandf;
    h_x_cublas[i] = h_x_cnc[i];
    h_y_cnc[i] = cncblasRandf;
    h_y_cublas[i] = h_y_cnc[i];
  }
  *h_alpha = cncblasRandf * 2 * PI;
  *h_c = std::cos(*h_alpha);
  *h_s = std::sin(*h_alpha);
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(float), cudaMemcpyHostToDevice));

  // Perform rot using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSrot(handle, N, d_x_cublas, 1, d_y_cublas, 1, h_c, h_s);

  // Perform rot using cncblas
  cncblasSrot(N, d_x_cnc, d_y_cnc, h_alpha);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_x_cnc[i], h_x_cublas[i]);
    EXPECT_FLOAT_EQ(h_y_cnc[i], h_y_cublas[i]);
  }

  // Free the memory
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_x_cublas;
  delete[] h_y_cublas;
  delete h_alpha;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

TEST(rot, doublePrecision) {
  double *h_alpha, *h_c, *h_s;
  double *h_x_cnc, *h_y_cnc, *h_x_cublas, *h_y_cublas;
  double *d_x_cnc, *d_y_cnc, *d_x_cublas, *d_y_cublas;

  h_x_cnc = new double[N];
  h_y_cnc = new double[N];
  h_x_cublas = new double[N];
  h_y_cublas = new double[N];
  h_alpha = new double;
  h_c = new double;
  h_s = new double;
  checkCudaErrors(cudaMalloc(&d_x_cnc, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cnc, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_x_cublas, N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y_cublas, N * sizeof(double)));

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_x_cnc[i] = cncblasRand;
    h_x_cublas[i] = h_x_cnc[i];
    h_y_cnc[i] = cncblasRand;
    h_y_cublas[i] = h_y_cnc[i];
  }
  *h_alpha = cncblasRand * 2 * PI;
  *h_c = std::cos(*h_alpha);
  *h_s = std::sin(*h_alpha);
  checkCudaErrors(cudaMemcpy(d_x_cnc, h_x_cnc, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cnc, h_y_cnc, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x_cublas, h_x_cublas, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_cublas, h_y_cublas, N * sizeof(double), cudaMemcpyHostToDevice));

  // Perform rot using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDrot(handle, N, d_x_cublas, 1, d_y_cublas, 1, h_c, h_s);

  // Perform rot using cncblas
  cncblasDrot(N, d_x_cnc, d_y_cnc, h_alpha);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_x_cnc, d_x_cnc, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cnc, d_y_cnc, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_x_cublas, d_x_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Check the results
  for (int i = 0; i < N; i++) {
    EXPECT_DOUBLE_EQ(h_x_cnc[i], h_x_cublas[i]);
    EXPECT_DOUBLE_EQ(h_y_cnc[i], h_y_cublas[i]);
  }

  // Free the memory
  delete[] h_x_cnc;
  delete[] h_y_cnc;
  delete[] h_x_cublas;
  delete[] h_y_cublas;
  delete h_alpha;
  delete h_c;
  delete h_s;
  checkCudaErrors(cudaFree(d_x_cnc));
  checkCudaErrors(cudaFree(d_y_cnc));
  checkCudaErrors(cudaFree(d_x_cublas));
  checkCudaErrors(cudaFree(d_y_cublas));
}

/* -------------------- ROTG --------------------- */

TEST(rotg, singlePresion) {
  float *h_a_cnc, *h_b_cnc, *h_alpha_cnc;
  float *d_a_cnc, *d_b_cnc, *d_alpha_cnc;
  float *h_a_cublas, *h_b_cublas, *h_c_cublas, *h_s_cublas;

  h_a_cnc = new float;
  h_b_cnc = new float;
  h_alpha_cnc = new float;
  h_a_cublas = new float;
  h_b_cublas = new float;
  h_c_cublas = new float;
  h_s_cublas = new float;
  checkCudaErrors(cudaMalloc(&d_a_cnc, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_b_cnc, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_alpha_cnc, sizeof(float)));

  *h_a_cnc = cncblasRandf;
  *h_b_cnc = cncblasRandf;
  *h_a_cublas = *h_a_cnc;
  *h_b_cublas = *h_b_cnc;
  checkCudaErrors(cudaMemcpy(d_a_cnc, h_a_cnc, sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b_cnc, h_b_cnc, sizeof(float), cudaMemcpyHostToDevice));

  // Perform rotg using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSrotg(handle, h_a_cublas, h_b_cublas, h_c_cublas, h_s_cublas);

  // Perform rotg using cncblas
  cncblasSrotg(d_a_cnc, d_b_cnc, d_alpha_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_a_cnc, d_a_cnc, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_b_cnc, d_b_cnc, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_alpha_cnc, d_alpha_cnc, sizeof(float), cudaMemcpyDeviceToHost));

  // Check the results
  EXPECT_FLOAT_EQ(*h_a_cnc, *h_a_cublas);
  EXPECT_FLOAT_EQ(*h_b_cnc, *h_b_cublas);
  EXPECT_FLOAT_EQ(*h_alpha_cnc, std::atan2(*h_s_cublas, *h_c_cublas));
}

TEST(rotg, doublePresion) {
  double *h_a_cnc, *h_b_cnc, *h_alpha_cnc;
  double *d_a_cnc, *d_b_cnc, *d_alpha_cnc;
  double *h_a_cublas, *h_b_cublas, *h_c_cublas, *h_s_cublas;

  h_a_cnc = new double;
  h_b_cnc = new double;
  h_alpha_cnc = new double;
  h_a_cublas = new double;
  h_b_cublas = new double;
  h_c_cublas = new double;
  h_s_cublas = new double;
  checkCudaErrors(cudaMalloc(&d_a_cnc, sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_b_cnc, sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_alpha_cnc, sizeof(double)));

  *h_a_cnc = cncblasRand;
  *h_b_cnc = cncblasRand;
  *h_a_cublas = *h_a_cnc;
  *h_b_cublas = *h_b_cnc;
  checkCudaErrors(cudaMemcpy(d_a_cnc, h_a_cnc, sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b_cnc, h_b_cnc, sizeof(double), cudaMemcpyHostToDevice));

  // Perform rotg using cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDrotg(handle, h_a_cublas, h_b_cublas, h_c_cublas, h_s_cublas);

  // Perform rotg using cncblas
  cncblasDrotg(d_a_cnc, d_b_cnc, d_alpha_cnc);

  // Copy the results back
  checkCudaErrors(cudaMemcpy(h_a_cnc, d_a_cnc, sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_b_cnc, d_b_cnc, sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_alpha_cnc, d_alpha_cnc, sizeof(double), cudaMemcpyDeviceToHost));

  // Check the results
  EXPECT_DOUBLE_EQ(*h_a_cnc, *h_a_cublas);
  EXPECT_DOUBLE_EQ(*h_b_cnc, *h_b_cublas);
  EXPECT_DOUBLE_EQ(*h_alpha_cnc, std::atan2(*h_s_cublas, *h_c_cublas));
}
