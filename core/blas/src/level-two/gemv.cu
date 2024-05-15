#include "cncblas.h"
#include "src/helpers.cuh"

/* -------------------- KERNEL DECLARATION -------------------- */

__global__ void cncblasSgemvKernelN(const int lenx, const int leny,
                                    const float *alpha, const float *A, const float *x, float *y);
__global__ void cncblasSgemvKernelT(const int lenx, const int leny,
                                    const float *alpha, const float *A, const float *x, float *y);
__global__ void cncblasDgemvKernelN(const int lenx, const int leny,
                                    const double *alpha, const double *A, const double *x, double *y);
__global__ void cncblasDgemvKernelT(const int lenx, const int leny,
                                    const double *alpha, const double *A, const double *x, double *y);
__global__ void cncblasCgemvKernelN(const int lenx, const int leny,
                                    const cuComplex *alpha, const cuComplex *A, const cuComplex *x, cuComplex *y);
__global__ void cncblasCgemvKernelT_1(const int lenx, const int leny,
                                      const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                      cuComplex *phase1);
__global__ void cncblasCgemvKernelT_2(const int n, const cuComplex *phase1, cuComplex *y);
__global__ void cncblasCgemvKernelC_1(const int lenx, const int leny,
                                      const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                      cuComplex *phase1);
__global__ void cncblasCgemvKernelC_2(const int n, const cuComplex *phase1, cuComplex *y);
__global__ void cncblasZgemvKernelN(const int lenx, const int leny,
                                    const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                                    cuDoubleComplex *y);
__global__ void cncblasZgemvKernelT_1(const int lenx, const int leny,
                                      const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                                      cuDoubleComplex *y);
__global__ void cncblasZgemvKernelT_2(const int n, const cuDoubleComplex *phase1, cuDoubleComplex *y);
__global__ void cncblasZgemvKernelC_1(const int lenx, const int leny,
                                      const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                                      cuDoubleComplex *y);
__global__ void cncblasZgemvKernelC_2(const int n, const cuDoubleComplex *phase1, cuDoubleComplex *y);

/* -------------------- GEMV -------------------- */

const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;

void cncblasSgemv(cncblasOperation_t trans,
                  int m, int n,
                  const float *alpha, const float *A, const float *x,
                  const float *beta, float *y) {
  // Test for invalid parameters
  gemvParamErrorCheck(m, n, alpha, A, x, beta, y);
  // Preprocess scalar pointers
  float *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  // quick return if possible
  if (m == 0 || n == 0 || (*h_alpha == 0 && *h_beta == 1)) {
    return;
  }

  // Set `lenx` and `leny` based on the value of `trans`.
  // `lenx` and `leny` are the lengths of the vectors `x` and `y` respectively.
  int lenx = 0, leny = 0;
  if (trans == CNCBLAS_OP_N) {
    lenx = n;
    leny = m;
  } else {
    lenx = m;
    leny = n;
  }

  // First form y = beta * y
  if (*h_beta != 1)
    cncblasSscal(leny, d_beta, y);
  if (*h_alpha == 0) return;

  // Form y = alpha * op(A) * x + y
  if (trans == CNCBLAS_OP_N) {
    // Form y = alpha * A * x + y
    int GRID_SIZE = m;
    cncblasSgemvKernelN<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, y);
  } else if (trans == CNCBLAS_OP_T) {
    // Form y = alpha * A^T * x + y
    dim3 GRID_SIZE((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cncblasSgemvKernelT<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, y);
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }

  // Free memory
  free(h_alpha);
  free(h_beta);
  checkCudaErrors(cudaFree(d_alpha));
  checkCudaErrors(cudaFree(d_beta));
}

void cncblasDgemv(cncblasOperation_t trans,
                  int m, int n,
                  const double *alpha, const double *A, const double *x,
                  const double *beta, double *y) {
  // Test for invalid parameters
  gemvParamErrorCheck(m, n, alpha, A, x, beta, y);
  // Preprocess scalar pointers
  double *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  // quick return if possible
  if (m == 0 || n == 0 || (*h_alpha == 0 && *h_beta == 1)) {
    return;
  }

  // Set `lenx` and `leny` based on the value of `trans`.
  // `lenx` and `leny` are the lengths of the vectors `x` and `y` respectively.
  int lenx = 0, leny = 0;
  if (trans == CNCBLAS_OP_N) {
    lenx = n;
    leny = m;
  } else {
    lenx = m;
    leny = n;
  }

  // First form y = beta * y
  if (*h_beta != 1)
    cncblasDscal(leny, d_beta, y);
  if (*h_alpha == 0) return;

  // Form y = alpha * op(A) * x + y
  if (trans == CNCBLAS_OP_N) {
    // Form y = alpha * A * x + y
    int GRID_SIZE = m;
    cncblasDgemvKernelN<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, y);
  } else if (trans == CNCBLAS_OP_T) {
    // Form y = alpha * A^T * x + y
    dim3 GRID_SIZE((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cncblasDgemvKernelT<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, y);
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }

  // Free memory
  free(h_alpha);
  free(h_beta);
  checkCudaErrors(cudaFree(d_alpha));
  checkCudaErrors(cudaFree(d_beta));
}

void cncblasCgemv(cncblasOperation_t trans,
                  int m, int n,
                  const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                  const cuComplex *beta, cuComplex *y) {
  // Test for invalid parameters
  gemvParamErrorCheck(m, n, alpha, A, x, beta, y);
  // Preprocess scalar pointers
  cuComplex *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  cuComplex one = make_cuComplex(1, 0);
  cuComplex zero = make_cuComplex(0, 0);
  // quick return if possible
  if (m == 0 || n == 0 || (cncblasComplexIsEqual(h_alpha, &zero) && cncblasComplexIsEqual(h_beta, &one))) {
    return;
  }

  // Set `lenx` and `leny` based on the value of `trans`.
  // `lenx` and `leny` are the lengths of the vectors `x` and `y` respectively.
  int lenx = 0, leny = 0;
  if (trans == CNCBLAS_OP_N) {
    lenx = n;
    leny = m;
  } else {
    lenx = m;
    leny = n;
  }

  // First form y = beta * y
  if (!cncblasComplexIsEqual(h_beta, &one))
    cncblasCscal(leny, d_beta, y);
  if (cncblasComplexIsEqual(h_alpha, &zero)) return;

  // Form y = alpha * op(A) * x + y
  if (trans == CNCBLAS_OP_N) {
    // Form y = alpha * A * x + y
    int GRID_SIZE = m;
    cncblasCgemvKernelN<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, y);
  } else if (trans == CNCBLAS_OP_T) {
    // Form y = alpha * A^T * x + y
    dim3 GRID_SIZE((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cuComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, leny * GRID_SIZE.y * sizeof(cuComplex)));
    cncblasCgemvKernelT_1<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, phase1);
    cncblasCgemvKernelT_2<<<leny, WARP_SIZE>>>(GRID_SIZE.y, phase1, y);
  } else if (trans == CNCBLAS_OP_C) {
    // Form y = alpha * A^H * x + y
    dim3 GRID_SIZE((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cuComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, leny * GRID_SIZE.y * sizeof(cuComplex)));
    cncblasCgemvKernelC_1<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, phase1);
    cncblasCgemvKernelC_2<<<leny, WARP_SIZE>>>(GRID_SIZE.y, phase1, y);
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }
}

void cncblasZgemv(cncblasOperation_t trans,
                  int m, int n,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                  const cuDoubleComplex *beta, cuDoubleComplex *y) {
  // Test for invalid parameters
  gemvParamErrorCheck(m, n, alpha, A, x, beta, y);
  // Preprocess scalar pointers
  cuDoubleComplex *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  cuDoubleComplex one = make_cuDoubleComplex(1, 0);
  cuDoubleComplex zero = make_cuDoubleComplex(0, 0);
  // quick return if possible
  if (m == 0 || n == 0 || (cncblasComplexIsEqual(h_alpha, &zero) && cncblasComplexIsEqual(h_beta, &one))) {
    return;
  }

  // Set `lenx` and `leny` based on the value of `trans`.
  // `lenx` and `leny` are the lengths of the vectors `x` and `y` respectively.
  int lenx = 0, leny = 0;
  if (trans == CNCBLAS_OP_N) {
    lenx = n;
    leny = m;
  } else {
    lenx = m;
    leny = n;
  }

  // First form y = beta * y
  if (!cncblasComplexIsEqual(h_beta, &one))
    cncblasZscal(leny, d_beta, y);
  if (cncblasComplexIsEqual(h_alpha, &zero)) return;

  // Form y = alpha * op(A) * x + y
  if (trans == CNCBLAS_OP_N) {
    // Form y = alpha * A * x + y
    int GRID_SIZE = m;
    cncblasZgemvKernelN<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, y);
  } else if (trans == CNCBLAS_OP_T) {
    // Form y = alpha * A^T * x + y
    dim3 GRID_SIZE((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cuDoubleComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, leny * GRID_SIZE.y * sizeof(cuDoubleComplex)));
    cncblasZgemvKernelT_1<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, phase1);
    cncblasZgemvKernelT_2<<<leny, WARP_SIZE>>>(GRID_SIZE.y, phase1, y);
  } else if (trans == CNCBLAS_OP_C) {
    // Form y = alpha * A^H * x + y
    dim3 GRID_SIZE((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cuDoubleComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, leny * GRID_SIZE.y * sizeof(cuDoubleComplex)));
    cncblasZgemvKernelC_1<<<GRID_SIZE, BLOCK_SIZE>>>(lenx, leny, d_alpha, A, x, phase1);
    cncblasZgemvKernelC_2<<<leny, WARP_SIZE>>>(GRID_SIZE.y, phase1, y);
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }
}

/* -------------------- KERNEL DEFINITION -------------------- */

__device__ void cncblasSgemvWarpRdN(volatile float *sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasSgemvKernelN(const int lenx, const int leny,
                                    const float *alpha, const float *A, const float *x, float *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ float sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  __syncthreads();
  int i = tid;
  while (i < lenx) {
    sdata[tid] += A[row * lenx + i] * x[i];
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasSgemvWarpRdN(sdata, tid);

  if (tid == 0) y[blockIdx.x] += *alpha * sdata[0];
}

__global__ void cncblasSgemvKernelT(const int lenx, const int leny,
                                    const float *alpha, const float *A, const float *x, float *y) {
  int grow = blockIdx.y * BLOCK_SIZE;
  int gcol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ float sdata[BLOCK_SIZE];
  sdata[threadIdx.x] = (grow + threadIdx.x < lenx) ? x[grow + threadIdx.x] : 0;
  __syncthreads();

  float temp = 0;
  for (int row = 0; row < BLOCK_SIZE; row++) {
    if (grow + row < lenx && gcol < leny) {
      temp += A[(grow + row) * leny + gcol] * sdata[row];
    }
  }

  if (gcol < leny)
    atomicAdd(&y[gcol], *alpha * temp);
}

__device__ void cncblasDgemvWarpRdN(volatile double *sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void cncblasDgemvKernelN(const int lenx, const int leny,
                                    const double *alpha, const double *A, const double *x, double *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ double sdata[BLOCK_SIZE];
  sdata[tid] = 0;
  __syncthreads();
  int i = tid;
  while (i < lenx) {
    sdata[tid] += A[row * lenx + i] * x[i];
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  __syncthreads();
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  __syncthreads();
  if (tid < 32) cncblasDgemvWarpRdN(sdata, tid);

  if (tid == 0) y[blockIdx.x] += *alpha * sdata[0];
}

__global__ void cncblasDgemvKernelT(const int lenx, const int leny,
                                    const double *alpha, const double *A, const double *x, double *y) {
  int grow = blockIdx.y * BLOCK_SIZE;
  int gcol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ double sdata[BLOCK_SIZE];
  sdata[threadIdx.x] = (grow + threadIdx.x < lenx) ? x[grow + threadIdx.x] : 0;
  __syncthreads();

  double temp = 0;
  for (int row = 0; row < BLOCK_SIZE; row++) {
    if (grow + row < lenx && gcol < leny) {
      temp += A[(grow + row) * leny + gcol] * sdata[row];
    }
  }

  if (gcol < leny)
    atomicAdd(&y[gcol], *alpha * temp);
}

__device__ void cncblasCgemvWarpRdN(volatile cuComplex *sdata, int tid) {
  volatile cuComplex *temp = sdata + tid;
  cncblasCVaddf(temp, temp + 32);
  cncblasCVaddf(temp, temp + 16);
  cncblasCVaddf(temp, temp + 8);
  cncblasCVaddf(temp, temp + 4);
  cncblasCVaddf(temp, temp + 2);
  cncblasCVaddf(temp, temp + 1);
}

__global__ void cncblasCgemvKernelN(const int lenx, const int leny,
                                    const cuComplex *alpha, const cuComplex *A, const cuComplex *x, cuComplex *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ cuComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuComplex(0, 0);
  __syncthreads();
  int i = tid;
  while (i < lenx) {
    sdata[tid] = cuCaddf(sdata[tid], cuCmulf(A[row * lenx + i], x[i]));
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  if (tid < 128) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCaddf(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasCgemvWarpRdN(sdata, tid);

  if (tid == 0) y[blockIdx.x] = cuCaddf(y[blockIdx.x], cuCmulf(*alpha, sdata[0]));
}

__global__ void cncblasCgemvKernelT_1(const int lenx, const int leny,
                                      const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                      cuComplex *phase1) {
  int grow = blockIdx.y * BLOCK_SIZE;
  int gcol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ cuComplex sdata[BLOCK_SIZE];
  sdata[threadIdx.x] = (grow + threadIdx.x < lenx) ? x[grow + threadIdx.x] : make_cuComplex(0, 0);
  __syncthreads();

  cuComplex temp = make_cuComplex(0, 0);
  for (int row = 0; row < BLOCK_SIZE; row++) {
    if (grow + row < lenx && gcol < leny) {
      temp = cuCaddf(temp, cuCmulf(A[(grow + row) * leny + gcol], sdata[row]));
    }
  }

  if (gcol < leny)
    phase1[gcol * gridDim.y + blockIdx.y] = cuCmulf(*alpha, temp);
}

__device__ void cncblasCgemvWarpRdT(volatile cuComplex *sdata, int tid) {
  volatile cuComplex *temp = sdata + tid;
  cncblasCVaddf(temp, temp + 16);
  cncblasCVaddf(temp, temp + 8);
  cncblasCVaddf(temp, temp + 4);
  cncblasCVaddf(temp, temp + 2);
  cncblasCVaddf(temp, temp + 1);
}

__global__ void cncblasCgemvKernelT_2(const int n, const cuComplex *phase1, cuComplex *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ cuComplex sdata[WARP_SIZE];
  sdata[tid] = make_cuComplex(0, 0);
  __syncthreads();
  int i = tid;
  while (i < n) {
    sdata[tid] = cuCaddf(sdata[tid], phase1[row * n + i]);
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  cncblasCgemvWarpRdT(sdata, tid);

  if (tid == 0) y[row] = cuCaddf(y[row], sdata[0]);
}

__global__ void cncblasCgemvKernelC_1(const int lenx, const int leny,
                                      const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                      cuComplex *phase1) {
  int grow = blockIdx.y * BLOCK_SIZE;
  int gcol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ cuComplex sdata[BLOCK_SIZE];
  sdata[threadIdx.x] = (grow + threadIdx.x < lenx) ? x[grow + threadIdx.x] : make_cuComplex(0, 0);
  __syncthreads();

  cuComplex temp = make_cuComplex(0, 0);
  for (int row = 0; row < BLOCK_SIZE; row++) {
    if (grow + row < lenx && gcol < leny) {
      temp = cuCaddf(temp, cuCmulf(cuConjf(A[(grow + row) * leny + gcol]), sdata[row]));
    }
  }

  if (gcol < leny)
    phase1[gcol * gridDim.y + blockIdx.y] = cuCmulf(*alpha, temp);
}

__global__ void cncblasCgemvKernelC_2(const int n, const cuComplex *phase1, cuComplex *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ cuComplex sdata[WARP_SIZE];
  sdata[tid] = make_cuComplex(0, 0);
  __syncthreads();
  int i = tid;
  while (i < n) {
    sdata[tid] = cuCaddf(sdata[tid], phase1[row * n + i]);
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  cncblasCgemvWarpRdT(sdata, tid);

  if (tid == 0) y[row] = cuCaddf(y[row], sdata[0]);
}

__device__ void cncblasZgemvWarpRdN(volatile cuDoubleComplex *sdata, int tid) {
  volatile cuDoubleComplex *temp = sdata + tid;
  cncblasZVadd(temp, temp + 32);
  cncblasZVadd(temp, temp + 16);
  cncblasZVadd(temp, temp + 8);
  cncblasZVadd(temp, temp + 4);
  cncblasZVadd(temp, temp + 2);
  cncblasZVadd(temp, temp + 1);
}

__global__ void cncblasZgemvKernelN(const int lenx, const int leny,
                                    const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                                    cuDoubleComplex *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ cuDoubleComplex sdata[BLOCK_SIZE];
  sdata[tid] = make_cuDoubleComplex(0, 0);
  __syncthreads();
  int i = tid;
  while (i < lenx) {
    sdata[tid] = cuCadd(sdata[tid], cuCmul(A[row * lenx + i], x[i]));
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  if (tid < 128) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 128]);
  __syncthreads();
  if (tid < 64) sdata[tid] = cuCadd(sdata[tid], sdata[tid + 64]);
  __syncthreads();
  if (tid < 32) cncblasZgemvWarpRdN(sdata, tid);

  if (tid == 0) y[blockIdx.x] = cuCadd(y[blockIdx.x], cuCmul(*alpha, sdata[0]));
}

__global__ void cncblasZgemvKernelT_1(const int lenx, const int leny,
                                      const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                                      cuDoubleComplex *phase1) {
  int grow = blockIdx.y * BLOCK_SIZE;
  int gcol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ cuDoubleComplex sdata[BLOCK_SIZE];
  sdata[threadIdx.x] = (grow + threadIdx.x < lenx) ? x[grow + threadIdx.x] : make_cuDoubleComplex(0, 0);
  __syncthreads();

  cuDoubleComplex temp = make_cuDoubleComplex(0, 0);
  for (int row = 0; row < BLOCK_SIZE; row++) {
    if (grow + row < lenx && gcol < leny) {
      temp = cuCadd(temp, cuCmul(A[(grow + row) * leny + gcol], sdata[row]));
    }
  }

  if (gcol < leny)
    phase1[gcol * gridDim.y + blockIdx.y] = cuCmul(*alpha, temp);
}

__device__ void cncblasZgemvWarpRdT(volatile cuDoubleComplex *sdata, int tid) {
  volatile cuDoubleComplex *temp = sdata + tid;
  cncblasZVadd(temp, temp + 16);
  cncblasZVadd(temp, temp + 8);
  cncblasZVadd(temp, temp + 4);
  cncblasZVadd(temp, temp + 2);
  cncblasZVadd(temp, temp + 1);
}

__global__ void cncblasZgemvKernelT_2(const int n, const cuDoubleComplex *phase1, cuDoubleComplex *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ cuDoubleComplex sdata[WARP_SIZE];
  sdata[tid] = make_cuDoubleComplex(0, 0);
  __syncthreads();
  int i = tid;
  while (i < n) {
    sdata[tid] = cuCadd(sdata[tid], phase1[row * n + i]);
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  cncblasZgemvWarpRdT(sdata, tid);

  if (tid == 0) y[row] = cuCadd(y[row], sdata[0]);
}

__global__ void cncblasZgemvKernelC_1(const int lenx, const int leny,
                                      const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                                      cuDoubleComplex *phase1) {
  int grow = blockIdx.y * BLOCK_SIZE;
  int gcol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ cuDoubleComplex sdata[BLOCK_SIZE];
  sdata[threadIdx.x] = (grow + threadIdx.x < lenx) ? x[grow + threadIdx.x] : make_cuDoubleComplex(0, 0);
  __syncthreads();

  cuDoubleComplex temp = make_cuDoubleComplex(0, 0);
  for (int row = 0; row < BLOCK_SIZE; row++) {
    if (grow + row < lenx && gcol < leny) {
      temp = cuCadd(temp, cuCmul(cuConj(A[(grow + row) * leny + gcol]), sdata[row]));
    }
  }

  if (gcol < leny)
    phase1[gcol * gridDim.y + blockIdx.y] = cuCmul(*alpha, temp);
}

__global__ void cncblasZgemvKernelC_2(const int n, const cuDoubleComplex *phase1, cuDoubleComplex *y) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int stride = blockDim.x;

  __shared__ cuDoubleComplex sdata[WARP_SIZE];
  sdata[tid] = make_cuDoubleComplex(0, 0);
  __syncthreads();
  int i = tid;
  while (i < n) {
    sdata[tid] = cuCadd(sdata[tid], phase1[row * n + i]);
    i += stride;
  }
  __syncthreads();

  // Reduce the sum with loop unrolling
  cncblasZgemvWarpRdT(sdata, tid);

  if (tid == 0) y[row] = cuCadd(y[row], sdata[0]);
}

