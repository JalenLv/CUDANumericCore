#include <level_one.cuh>
#include "helpers.cuh"
#include <stdexcept>
#include <iostream>

/* -------------------- DOT -------------------- */

const int BLOCK_SIZE = 256;
const int GRID_SIZE = 256;

__device__ void cncblasSdotWarpReduce(volatile float *sdata, size_t tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasSdotKernel_1(size_t n, const float *x, const float *y, float *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ float sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += x[i] * y[i];
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasSdotWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

__global__ void cncblasSdotKernel_2(size_t n, const float *result_phase1, float *result_phase2) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ float sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += result_phase1[i];
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasSdotWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) *result_phase2 = sdata[0];
}

float cncblasSdot(size_t n, const float *x, const float *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // assign memory for result
  float *d_result_phaes1;
  cudaMalloc(&d_result_phaes1, GRID_SIZE * sizeof(float));
  float *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(float));

  // launch kernel
  cncblasSdotKernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, y, d_result_phaes1);
  cncblasSdotKernel_2<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phaes1, d_result_phase2);

  // save result
  float result;
  cudaMemcpy(&result, d_result_phase2, sizeof(float), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_result_phaes1);
  cudaFree(d_result_phase2);

  return result;
}

__device__ void cncblasDdotWarpReduce(volatile double *sdata, size_t tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasDdotKernel_1(size_t n, const double *x, const double *y, double *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ double sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += x[i] * y[i];
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasDdotWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

__global__ void cncblasDdotKernel_2(size_t n, const double *result_phase1, double *result_phase2) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ double sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  size_t i = idx;
  while (i < n) {
    sdata[tid] += result_phase1[i];
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasDdotWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) *result_phase2 = sdata[0];
}

double cncblasDdot(size_t n, const double *x, const double *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // assign memory for result
  double *d_result_phaes1;
  cudaMalloc(&d_result_phaes1, GRID_SIZE * sizeof(double));
  double *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(double));

  // launch kernel
  cncblasDdotKernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, y, d_result_phaes1);
  cncblasDdotKernel_2<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phaes1, d_result_phase2);

  // save result
  double result;
  cudaMemcpy(&result, d_result_phase2, sizeof(double), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_result_phaes1);
  cudaFree(d_result_phase2);

  return result;
}

__device__ void cncblasCdotuWarpReduce(volatile cuComplex *sdata, size_t tid) {
  volatile cuComplex *temp = sdata + tid;
  cncblasCVaddf(temp, temp + 32);
  cncblasCVaddf(temp, temp + 16);
  cncblasCVaddf(temp, temp + 8);
  cncblasCVaddf(temp, temp + 4);
  cncblasCVaddf(temp, temp + 2);
  cncblasCVaddf(temp, temp + 1);
}

__global__ void cncblasCdotuKernel_1(size_t n, const cuComplex *x, const cuComplex *y, cuComplex *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCaddf(sdata[tid], cuCmulf(x[i], y[i]));
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasCdotuWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

__global__ void cncblasCdotuKernel_2(size_t n, const cuComplex *result_phase1, cuComplex *result_phase2) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCaddf(sdata[tid], result_phase1[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasCdotuWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) *result_phase2 = sdata[0];
}

cuComplex cncblasCdotu(size_t n, const cuComplex *x, const cuComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // assign memory for result
  cuComplex *d_result_phaes1;
  cudaMalloc(&d_result_phaes1, GRID_SIZE * sizeof(cuComplex));
  cuComplex *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(cuComplex));

  // launch kernel
  cncblasCdotuKernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, y, d_result_phaes1);
  cncblasCdotuKernel_2<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phaes1, d_result_phase2);

  // save result
  cuComplex result;
  cudaMemcpy(&result, d_result_phase2, sizeof(cuComplex), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_result_phaes1);
  cudaFree(d_result_phase2);

  return result;
}

__device__ void cncblasZdotuWarpReduce(volatile cuDoubleComplex *sdata, size_t tid) {
  volatile cuDoubleComplex *temp = sdata + tid;
  cncblasZVadd(temp, temp + 32);
  cncblasZVadd(temp, temp + 16);
  cncblasZVadd(temp, temp + 8);
  cncblasZVadd(temp, temp + 4);
  cncblasZVadd(temp, temp + 2);
  cncblasZVadd(temp, temp + 1);
}

__global__ void
cncblasZdotuKernel_1(size_t n, const cuDoubleComplex *x, const cuDoubleComplex *y, cuDoubleComplex *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuDoubleComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuDoubleComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCadd(sdata[tid], cuCmul(x[i], y[i]));
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasZdotuWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

__global__ void cncblasZdotuKernel_2(size_t n, const cuDoubleComplex *result_phase1, cuDoubleComplex *result_phase2) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuDoubleComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuDoubleComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCadd(sdata[tid], result_phase1[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasZdotuWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) *result_phase2 = sdata[0];
}

cuDoubleComplex cncblasZdotu(size_t n, const cuDoubleComplex *x, const cuDoubleComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // assign memory for result
  cuDoubleComplex *d_result_phaes1;
  cudaMalloc(&d_result_phaes1, GRID_SIZE * sizeof(cuDoubleComplex));
  cuDoubleComplex *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(cuDoubleComplex));

  // launch kernel
  cncblasZdotuKernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, y, d_result_phaes1);
  cncblasZdotuKernel_2<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phaes1, d_result_phase2);

  // save result
  cuDoubleComplex result;
  cudaMemcpy(&result, d_result_phase2, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_result_phaes1);
  cudaFree(d_result_phase2);

  return result;
}

__device__ void cncblasCdotcWarpReduce(volatile cuComplex *sdata, size_t tid) {
  volatile cuComplex *temp = sdata + tid;
  cncblasCVaddf(temp, temp + 32);
  cncblasCVaddf(temp, temp + 16);
  cncblasCVaddf(temp, temp + 8);
  cncblasCVaddf(temp, temp + 4);
  cncblasCVaddf(temp, temp + 2);
  cncblasCVaddf(temp, temp + 1);
}

__global__ void cncblasCdotcKernel_1(size_t n, const cuComplex *x, const cuComplex *y, cuComplex *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCaddf(sdata[tid], cuCmulf(cuConjf(x[i]), y[i]));
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasCdotcWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

__global__ void cncblasCdotcKernel_2(size_t n, const cuComplex *result_phase1, cuComplex *result_phase2) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCaddf(sdata[tid], result_phase1[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasCdotcWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) *result_phase2 = sdata[0];
}

cuComplex cncblasCdotc(size_t n, const cuComplex *x, const cuComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // assign memory for result
  cuComplex *d_result_phaes1;
  cudaMalloc(&d_result_phaes1, GRID_SIZE * sizeof(cuComplex));
  cuComplex *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(cuComplex));

  // launch kernel
  cncblasCdotcKernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, y, d_result_phaes1);
  cncblasCdotcKernel_2<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phaes1, d_result_phase2);

  // save result
  cuComplex result;
  cudaMemcpy(&result, d_result_phase2, sizeof(cuComplex), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_result_phaes1);
  cudaFree(d_result_phase2);

  return result;
}

__device__ void cncblasZdotcWarpReduce(volatile cuDoubleComplex *sdata, size_t tid) {
  volatile cuDoubleComplex *temp = sdata + tid;
  cncblasZVadd(temp, temp + 32);
  cncblasZVadd(temp, temp + 16);
  cncblasZVadd(temp, temp + 8);
  cncblasZVadd(temp, temp + 4);
  cncblasZVadd(temp, temp + 2);
  cncblasZVadd(temp, temp + 1);
}

__global__ void
cncblasZdotcKernel_1(size_t n, const cuDoubleComplex *x, const cuDoubleComplex *y, cuDoubleComplex *result) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuDoubleComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuDoubleComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCadd(sdata[tid], cuCmul(cuConj(x[i]), y[i]));
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasZdotcWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) result[blockIdx.x] = sdata[0];
}

__global__ void cncblasZdotcKernel_2(size_t n, const cuDoubleComplex *result_phase1, cuDoubleComplex *result_phase2) {
  // Calculate index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // Shared memory
  __shared__ cuDoubleComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuDoubleComplex(0, 0);
  size_t i = idx;
  while (i < n) {
    sdata[tid] = cuCadd(sdata[tid], result_phase1[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Reduction with loop unrolling
  if (tid < 128) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasZdotcWarpReduce(sdata, tid);

  // Save result
  if (tid == 0) *result_phase2 = sdata[0];
}

cuDoubleComplex cncblasZdotc(size_t n, const cuDoubleComplex *x, const cuDoubleComplex *y) {
  // check if x and y are not null
  try {
    if (x == nullptr || y == nullptr) {
      throw std::invalid_argument("x and y must not be null");
    }
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // assign memory for result
  cuDoubleComplex *d_result_phaes1;
  cudaMalloc(&d_result_phaes1, GRID_SIZE * sizeof(cuDoubleComplex));
  cuDoubleComplex *d_result_phase2;
  cudaMalloc(&d_result_phase2, sizeof(cuDoubleComplex));

  // launch kernel
  cncblasZdotcKernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(n, x, y, d_result_phaes1);
  cncblasZdotcKernel_2<<<1, GRID_SIZE>>>(GRID_SIZE, d_result_phaes1, d_result_phase2);

  // save result
  cuDoubleComplex result;
  cudaMemcpy(&result, d_result_phase2, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_result_phaes1);
  cudaFree(d_result_phase2);

  return result;
}
