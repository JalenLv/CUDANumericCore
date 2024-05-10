#include "../cncblas.h"
#include "helpers.cuh"
#include <stdexcept>
#include <iostream>

#ifndef MAX_THREADS_PER_SM
#define MAX_THREADS_PER_SM 2048
#endif // !MAX_THREADS_PER_SM

/* ----------------------------- AMAX ----------------------------- */

const int BLOCK_SIZE = 256;
const int GRID_SIZE = 256;

__global__ void cncblasSamaxKernel(size_t n, const float *x, float *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ float sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = 0;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] < fabsf(x[i])) {
      sdata[tid] = fabsf(x[i]);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] < sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasSamax(size_t n, const float *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  float *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  float *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(float)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasSamaxKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasSamaxKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the maximum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}

__global__ void cncblasDamaxKernel(size_t n, const double *x, double *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ double sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = 0;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] < fabs(x[i])) {
      sdata[tid] = fabs(x[i]);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] < sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasDamax(size_t n, const double *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  double *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  double *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(double)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasDamaxKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasDamaxKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the maximum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}

__global__ void cncblasCamaxKernel(size_t n, const cuComplex *x, float *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ float sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = 0;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] < cncblasCmag(x + i)) {
      sdata[tid] = cncblasCmag(x + i);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] < sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasCamax(size_t n, const cuComplex *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  float *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  float *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(float)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasCamaxKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasSamaxKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the maximum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}

__global__ void cncblasZamaxKernel(size_t n, const cuDoubleComplex *x, double *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ double sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = 0;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] < cncblasZmag(x + i)) {
      sdata[tid] = cncblasZmag(x + i);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] < sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasZamax(size_t n, const cuDoubleComplex *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  double *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  double *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(double)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasZamaxKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasDamaxKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the maximum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}

/* ----------------------------- AMIN ----------------------------- */

__global__ void cncblasSaminKernel(size_t n, const float *x, float *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ float sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = INFINITY;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] > fabsf(x[i])) {
      sdata[tid] = fabsf(x[i]);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] > sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasSamin(size_t n, const float *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  float *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  float *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(float)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasSaminKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasSaminKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the minimum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}

__global__ void cncblasDaminKernel(size_t n, const double *x, double *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ double sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = INFINITY;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] > fabs(x[i])) {
      sdata[tid] = fabs(x[i]);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] > sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasDamin(size_t n, const double *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  double *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  double *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(double)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasDaminKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasDamaxKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the minimum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}

__global__ void cncblasCaminKernel(size_t n, const cuComplex *x, float *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ float sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = INFINITY;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] > cncblasCmag(x + i)) {
      sdata[tid] = cncblasCmag(x + i);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] > sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasCamin(size_t n, const cuComplex *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  float *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  float *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(float)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasCaminKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasSaminKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the minimum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}

__global__ void cncblasZaminKernel(size_t n, const cuDoubleComplex *x, double *result_data, size_t *result_index) {
  // Calculate the thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory for the block
  __shared__ double sdata[BLOCK_SIZE];
  __shared__ size_t sindex[BLOCK_SIZE];
  sdata[tid] = INFINITY;
  sindex[tid] = 0;
  size_t i = idx;
  while (i < n) {
    if (sdata[tid] > cncblasZmag(x + i)) {
      sdata[tid] = cncblasZmag(x + i);
      sindex[tid] = i;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Perform the reduction
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] > sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sindex[tid] = sindex[tid + s];
      }
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    result_data[blockIdx.x] = sdata[0];
    result_index[blockIdx.x] = sindex[0];
  }
}

size_t cncblasZamin(size_t n, const cuDoubleComplex *x) {
  // Check for invalid arguments
  try {
    if (x == nullptr) throw std::invalid_argument("x must be a device pointer");
    if (n <= 0) throw std::invalid_argument("n must be greater than 0");
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid Argument: " << e.what() << std::endl;
    return 0;
  }

  // Allocate memory for the result
  double *d_data_phase1;
  size_t *d_index_phase1;
  size_t *h_index_phase1 = new size_t[GRID_SIZE];
  checkCudaErrors(cudaMalloc(&d_data_phase1, GRID_SIZE * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_index_phase1, GRID_SIZE * sizeof(size_t)));

  double *d_data_phase2;
  checkCudaErrors(cudaMalloc(&d_data_phase2, 1 * sizeof(double)));
  size_t *h_index_phase2 = new size_t;
  size_t *d_index_phase2;
  checkCudaErrors(cudaMalloc(&d_index_phase2, 1 * sizeof(size_t)));

  // Launch the kernel
  cncblasZaminKernel<<<GRID_SIZE, BLOCK_SIZE>>>
          (n, x, d_data_phase1, d_index_phase1);
  cncblasDamaxKernel<<<1, GRID_SIZE>>>
          (GRID_SIZE, d_data_phase1, d_data_phase2, d_index_phase2);

  // Copy the result back to the host
  checkCudaErrors(cudaMemcpy(h_index_phase1, d_index_phase1,
                             GRID_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_index_phase2, d_index_phase2,
                             1 * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Find the minimum element
  size_t result = h_index_phase1[h_index_phase2[0]];

  // Free memory
  checkCudaErrors(cudaFree(d_data_phase1));
  checkCudaErrors(cudaFree(d_index_phase1));
  checkCudaErrors(cudaFree(d_data_phase2));
  checkCudaErrors(cudaFree(d_index_phase2));
  delete[] h_index_phase1;
  delete h_index_phase2;

  return result;
}