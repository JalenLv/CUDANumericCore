#include <../cncblas.h>
#include "helpers.cuh"
#include <stdexcept>
#include <iostream>

/* ------------------------- ASUM ------------------------- */

const int BLOCK_SIZE = 256;
const int GRID_SIZE = 256;

__device__ void cncblasSasumWarpReduce(volatile float *sdata, size_t tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasSasumKernel(size_t n, const float *x, float *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ float sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += fabsf(x[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasSasumWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

float cncblasSasum(size_t n, const float *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for result
  float *d_result_phase1;
  cudaMalloc(&d_result_phase1, BLOCK_SIZE * sizeof(float));
  float *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(float));
  float *h_result_phase2 = new float;

  // Launch kernel
  cncblasSasumKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, d_result_phase1);
  cncblasSasumKernel<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phase1, d_result_phase2);

  // Copy result back to host
  cudaMemcpy(h_result_phase2, d_result_phase2, sizeof(float), cudaMemcpyDeviceToHost);

  // Save result
  float result = *h_result_phase2;

  // Free memory
  cudaFree(d_result_phase1);
  cudaFree(d_result_phase2);
  delete h_result_phase2;

  return result;
}

__device__ void cncblasDasumWarpReduce(volatile double *sdata, size_t tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasDasumKernel(size_t n, const double *x, double *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ double sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += fabsf(x[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasDasumWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

double cncblasDasum(size_t n, const double *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for result
  double *d_result_phase1;
  cudaMalloc(&d_result_phase1, BLOCK_SIZE * sizeof(double));
  double *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(double));
  double *h_result_phase2 = new double;

  // Launch kernel
  cncblasDasumKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, d_result_phase1);
  cncblasDasumKernel<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phase1, d_result_phase2);

  // Copy result back to host
  cudaMemcpy(h_result_phase2, d_result_phase2, sizeof(double), cudaMemcpyDeviceToHost);

  // Save result
  double result = *h_result_phase2;

  // Free memory
  cudaFree(d_result_phase1);
  cudaFree(d_result_phase2);
  delete h_result_phase2;

  return result;
}

__device__ void cncblasCasumWarpReduce(volatile float *sdata, size_t tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasCasumKernel(size_t n, const cuComplex *x, float *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ float sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += cncblasCmag(x + i);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasSasumWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

float cncblasCasum(size_t n, const cuComplex *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for result
  float *d_result_phase1;
  cudaMalloc(&d_result_phase1, BLOCK_SIZE * sizeof(float));
  float *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(float));
  float *h_result_phase2 = new float;

  // Launch kernel
  cncblasCasumKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, d_result_phase1);
  cncblasSasumKernel<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phase1, d_result_phase2);

  // Copy result back to host
  cudaMemcpy(h_result_phase2, d_result_phase2, sizeof(float), cudaMemcpyDeviceToHost);

  // Save result
  float result = *h_result_phase2;

  // Free memory
  cudaFree(d_result_phase1);
  cudaFree(d_result_phase2);
  delete h_result_phase2;

  return result;
}

__device__ void cncblasZasumWarpReduce(volatile double *sdata, size_t tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasZasumKernel(size_t n, const cuDoubleComplex *x, double *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ double sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += cncblasZmag(x + i);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasDasumWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

double cncblasZasum(size_t n, const cuDoubleComplex *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for result
  double *d_result_phase1;
  cudaMalloc(&d_result_phase1, BLOCK_SIZE * sizeof(double));
  double *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(double));
  double *h_result_phase2 = new double;

  // Launch kernel
  cncblasZasumKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, d_result_phase1);
  cncblasDasumKernel<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phase1, d_result_phase2);

  // Copy result back to host
  cudaMemcpy(h_result_phase2, d_result_phase2, sizeof(double), cudaMemcpyDeviceToHost);

  // Save result
  double result = *h_result_phase2;

  // Free memory
  cudaFree(d_result_phase1);
  cudaFree(d_result_phase2);
  delete h_result_phase2;

  return result;
}
