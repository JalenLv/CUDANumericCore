#include "cncblas.h"
#include "src/helpers.cuh"

/* -------------------- KERNEL DECLARATION --------------------- */

__global__ void cncblasSgbmvKernelN(int m, int ku, int nColsA, int nRowsA,
                                    const float *alpha, const float *A, const float *x, float *y);
__global__ void cncblasSgbmvKernelT(int m, int ku, int nRowsA, int nColsA,
                                    const float *alpha, const float *A, const float *x, float *y);
__global__ void cncblasDgbmvKernelN(int m, int ku, int nColsA, int nRowsA,
                                    const double *alpha, const double *A, const double *x, double *y);
__global__ void cncblasDgbmvKernelT(int m, int ku, int nRowsA, int nColsA,
                                    const double *alpha, const double *A, const double *x, double *y);
__global__ void cncblasCgbmvKernelN_phase1(int m, int ku, int nColsA, int nRowsA,
                                           const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                           cuComplex *phase1);
__global__ void cncblasCgbmvKernelN_phase2(int m, const cuComplex *phase1, cuComplex *y);
__global__ void cncblasCgbmvKernelT_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                           cuComplex *phase1);
__global__ void cncblasCgbmvKernelT_phase2(int nColsA, const cuComplex *phase1, cuComplex *y);
__global__ void cncblasCgbmvKernelC_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                           cuComplex *phase1);
__global__ void cncblasCgbmvKernelC_phase2(int nColsA, const cuComplex *phase1, cuComplex *y);
__global__ void cncblasZgbmvKernelN_phase1(int m, int ku, int nColsA, int nRowsA,
                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                           const cuDoubleComplex *x,
                                           cuDoubleComplex *phase1);
__global__ void cncblasZgbmvKernelN_phase2(int m, const cuDoubleComplex *phase1, cuDoubleComplex *y);
__global__ void cncblasZgbmvKernelT_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                           const cuDoubleComplex *x,
                                           cuDoubleComplex *phase1);
__global__ void cncblasZgbmvKernelT_phase2(int nColsA, const cuDoubleComplex *phase1, cuDoubleComplex *y);
__global__ void cncblasZgbmvKernelC_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                           const cuDoubleComplex *x,
                                           cuDoubleComplex *phase1);
__global__ void cncblasZgbmvKernelC_phase2(int nColsA, const cuDoubleComplex *phase1, cuDoubleComplex *y);

/* -------------------- GBMV --------------------- */

const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;

void cncblasSgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const float *alpha, const float *A, const float *x,
                  const float *beta, float *y) {
  // Check the parameters
  gbmvParamErrorCheck(m, n, kl, ku, alpha, A, x, beta, y);
  if (trans == CNCBLAS_OP_C) {
    std::cerr << "CNCBLAS_OP_C is not supported" << std::endl;
    exit(1);
  }
  // Preprocess the scalar parameters
  float *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  // Quick return if possible
  if (m == 0 || n == 0 || (*h_alpha == 0 && *h_beta == 1)) {
    return;
  }

  // Set LENY, the length of the vectors x and y
  int leny;
  if (trans == CNCBLAS_OP_N) {
    leny = m;
  } else {
    leny = n;
  }

  // First, form y = beta * y
  if (*h_beta != 1) {
    cncblasSscal(leny, d_beta, y);
  }

  // Form y = alpha * A * x + y
  const int nColsA = cncblasMin(n, m + ku);
  const int nRowsA = ku + kl + 1;
  if (trans == CNCBLAS_OP_N) {
    const dim3 GRID_SIZE((m + BLOCK_SIZE - 1) / BLOCK_SIZE, 32);
    cncblasSgbmvKernelN<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nColsA, nRowsA, d_alpha, A, x, y);
  } else if (trans == CNCBLAS_OP_T) {
    const dim3 GRID_SIZE((nColsA + BLOCK_SIZE - 1) / BLOCK_SIZE, 32);
    cncblasSgbmvKernelT<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nRowsA, nColsA, d_alpha, A, x, y);
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }
}

void cncblasDgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const double *alpha, const double *A, const double *x,
                  const double *beta, double *y) {
  // Check the parameters
  gbmvParamErrorCheck(m, n, kl, ku, alpha, A, x, beta, y);
  if (trans == CNCBLAS_OP_C) {
    std::cerr << "CNCBLAS_OP_C is not supported" << std::endl;
    exit(1);
  }
  // Preprocess the scalar parameters
  double *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  // Quick return if possible
  if (m == 0 || n == 0 || (*h_alpha == 0 && *h_beta == 1)) {
    return;
  }

  // Set LENY, the length of the vectors x and y
  int leny;
  if (trans == CNCBLAS_OP_N) {
    leny = m;
  } else {
    leny = n;
  }

  // First, form y = beta * y
  if (*h_beta != 1) {
    cncblasDscal(leny, d_beta, y);
  }

  // Form y = alpha * A * x + y
  const int nColsA = cncblasMin(n, m + ku);
  const int nRowsA = ku + kl + 1;
  if (trans == CNCBLAS_OP_N) {
    const dim3 GRID_SIZE((m + BLOCK_SIZE - 1) / BLOCK_SIZE, 32);
    cncblasDgbmvKernelN<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nColsA, nRowsA, d_alpha, A, x, y);
  } else if (trans == CNCBLAS_OP_T) {
    const dim3 GRID_SIZE((nColsA + BLOCK_SIZE - 1) / BLOCK_SIZE, 32);
    cncblasDgbmvKernelT<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nRowsA, nColsA, d_alpha, A, x, y);
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }
}

void cncblasCgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                  const cuComplex *beta, cuComplex *y) {
  // Check the parameters
  gbmvParamErrorCheck(m, n, kl, ku, alpha, A, x, beta, y);
  // Preprocess the scalar parameters
  cuComplex *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  // Quick return if possible
  cuComplex zero = make_cuComplex(0.0f, 0.0f);
  cuComplex one = make_cuComplex(1.0f, 0.0f);
  if (m == 0 || n == 0 || (cncblasComplexIsEqual(h_alpha, &zero) && cncblasComplexIsEqual(h_beta, &one))) {
    return;
  }

  // Set LENY, the length of the vectors x and y
  int leny;
  if (trans == CNCBLAS_OP_N) {
    leny = m;
  } else {
    leny = n;
  }

  // First, form y = beta * y
  if (!cncblasComplexIsEqual(h_beta, &one)) {
    cncblasCscal(leny, d_beta, y);
  }

  // Form y = alpha * A * x + y
  const int nColsA = cncblasMin(n, m + ku);
  const int nRowsA = ku + kl + 1;
  if (trans == CNCBLAS_OP_N) {
    const dim3 GRID_SIZE((m + BLOCK_SIZE - 1) / BLOCK_SIZE, WARP_SIZE);

    cuComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, m * GRID_SIZE.y * sizeof(cuComplex)));

    cncblasCgbmvKernelN_phase1<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nColsA, nRowsA, d_alpha, A, x, phase1);
    cncblasCgbmvKernelN_phase2<<<GRID_SIZE.x, BLOCK_SIZE>>>(m, phase1, y);

    checkCudaErrors(cudaFree(phase1));
  } else if (trans == CNCBLAS_OP_T) {
    const dim3 GRID_SIZE((nColsA + BLOCK_SIZE - 1) / BLOCK_SIZE, WARP_SIZE);

    cuComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, nColsA * GRID_SIZE.y * sizeof(cuComplex)));

    cncblasCgbmvKernelT_phase1<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nRowsA, nColsA, d_alpha, A, x, phase1);
    cncblasCgbmvKernelT_phase2<<<GRID_SIZE.x, BLOCK_SIZE>>>(nColsA, phase1, y);

    checkCudaErrors(cudaFree(phase1));
  } else if (trans == CNCBLAS_OP_C) {
    const dim3 GRID_SIZE((nColsA + BLOCK_SIZE - 1) / BLOCK_SIZE, WARP_SIZE);

    cuComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, nColsA * GRID_SIZE.y * sizeof(cuComplex)));

    cncblasCgbmvKernelC_phase1<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nRowsA, nColsA, d_alpha, A, x, phase1);
    cncblasCgbmvKernelC_phase2<<<GRID_SIZE.x, BLOCK_SIZE>>>(nColsA, phase1, y);

    checkCudaErrors(cudaFree(phase1));
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }
}

void cncblasZgbmv(cncblasOperation_t trans,
                  int m, int n, int kl, int ku,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *x,
                  const cuDoubleComplex *beta, cuDoubleComplex *y) {
  // Check the parameters
  gbmvParamErrorCheck(m, n, kl, ku, alpha, A, x, beta, y);
  // Preprocess the scalar parameters
  cuDoubleComplex *h_alpha, *h_beta, *d_alpha, *d_beta;
  cncblasScalarPointerPreprocess(alpha, h_alpha, d_alpha);
  cncblasScalarPointerPreprocess(beta, h_beta, d_beta);

  // Quick return if possible
  cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  if (m == 0 || n == 0 || (cncblasComplexIsEqual(h_alpha, &zero) && cncblasComplexIsEqual(h_beta, &one))) {
    return;
  }

  // Set LENY, the length of the vectors x and y
  int leny;
  if (trans == CNCBLAS_OP_N) {
    leny = m;
  } else {
    leny = n;
  }

  // First, form y = beta * y
  if (!cncblasComplexIsEqual(h_beta, &one)) {
    cncblasZscal(leny, d_beta, y);
  }

  // Form y = alpha * A * x + y
  const int nColsA = cncblasMin(n, m + ku);
  const int nRowsA = ku + kl + 1;
  if (trans == CNCBLAS_OP_N) {
    const dim3 GRID_SIZE((m + BLOCK_SIZE - 1) / BLOCK_SIZE, WARP_SIZE);

    cuDoubleComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, m * GRID_SIZE.y * sizeof(cuDoubleComplex)));

    cncblasZgbmvKernelN_phase1<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nColsA, nRowsA, d_alpha, A, x, phase1);
    cncblasZgbmvKernelN_phase2<<<GRID_SIZE.x, BLOCK_SIZE>>>(m, phase1, y);

    checkCudaErrors(cudaFree(phase1));
  } else if (trans == CNCBLAS_OP_T) {
    const dim3 GRID_SIZE((nColsA + BLOCK_SIZE - 1) / BLOCK_SIZE, WARP_SIZE);

    cuDoubleComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, nColsA * GRID_SIZE.y * sizeof(cuDoubleComplex)));

    cncblasZgbmvKernelT_phase1<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nRowsA, nColsA, d_alpha, A, x, phase1);
    cncblasZgbmvKernelT_phase2<<<GRID_SIZE.x, BLOCK_SIZE>>>(nColsA, phase1, y);

    checkCudaErrors(cudaFree(phase1));
  } else if (trans == CNCBLAS_OP_C) {
    const dim3 GRID_SIZE((nColsA + BLOCK_SIZE - 1) / BLOCK_SIZE, WARP_SIZE);

    cuDoubleComplex *phase1;
    checkCudaErrors(cudaMalloc(&phase1, nColsA * GRID_SIZE.y * sizeof(cuDoubleComplex)));

    cncblasZgbmvKernelC_phase1<<<GRID_SIZE, BLOCK_SIZE>>>
            (m, ku, nRowsA, nColsA, d_alpha, A, x, phase1);
    cncblasZgbmvKernelC_phase2<<<GRID_SIZE.x, BLOCK_SIZE>>>(nColsA, phase1, y);

    checkCudaErrors(cudaFree(phase1));
  } else {
    std::cerr << "Invalid value for `trans`" << std::endl;
    exit(1);
  }
}

/* -------------------- KERNEL DEFINITION --------------------- */

__global__ void cncblasSgbmvKernelN(int m, int ku, int nColsA, int nRowsA,
                                    const float *alpha, const float *A, const float *x, float *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int offset = ku - row;
  int col = idx + offset;

  if (idx < m) {
    float temp = 0.0f;
    while (row < nRowsA && col >= 0) {
      if (col < nColsA && col >= 0) {
        temp += A[row * nColsA + col] * x[col];
      }
      row += gridDim.y;
      col -= gridDim.y;
    }

    atomicAdd(&y[idx], *alpha * temp);
  }
}

__global__ void cncblasSgbmvKernelT(int m, int ku, int nRowsA, int nColsA,
                                    const float *alpha, const float *A, const float *x, float *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int col = idx;
  int offset = row - ku;

  if (idx < nColsA) {
    float temp = 0.0f;
    while (row < nRowsA) {
      int i = col + offset;
      if (i >= 0 && i < m) {
        temp += A[row * nColsA + col] * x[i];
      }
      row += gridDim.y;
      offset += gridDim.y;
    }

    atomicAdd(&y[col], *alpha * temp);
  }
}

__global__ void cncblasDgbmvKernelN(int m, int ku, int nColsA, int nRowsA,
                                    const double *alpha, const double *A, const double *x, double *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int offset = ku - row;
  int col = idx + offset;

  if (idx < m) {
    double temp = 0.0f;
    while (row < nRowsA && col >= 0) {
      if (col < nColsA && col >= 0) {
        temp += A[row * nColsA + col] * x[col];
      }
      row += gridDim.y;
      col -= gridDim.y;
    }

    atomicAdd(&y[idx], *alpha * temp);
  }
}

__global__ void cncblasDgbmvKernelT(int m, int ku, int nRowsA, int nColsA,
                                    const double *alpha, const double *A, const double *x, double *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int col = idx;
  int offset = row - ku;

  if (idx < nColsA) {
    double temp = 0.0f;
    while (row < nRowsA) {
      int i = col + offset;
      if (i >= 0 && i < m) {
        temp += A[row * nColsA + col] * x[i];
      }
      row += gridDim.y;
      offset += gridDim.y;
    }

    atomicAdd(&y[col], *alpha * temp);
  }
}

__global__ void cncblasCgbmvKernelN_phase1(int m, int ku, int nColsA, int nRowsA,
                                           const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                           cuComplex *phase1) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int offset = ku - row;
  int col = idx + offset;

  if (tid < m) {
    cuComplex temp = make_cuComplex(0.0f, 0.0f);
    while (row < nRowsA && col >= 0) {
      if (col < nColsA && col >= 0) {
        temp = cuCaddf(temp, cuCmulf(A[row * nColsA + col], x[col]));
      }
      row += gridDim.y;
      col -= gridDim.y;
    }

    phase1[blockIdx.y * m + idx] = cuCmulf(*alpha, temp);
  }
}

__global__ void cncblasCgbmvKernelN_phase2(int m, const cuComplex *phase1, cuComplex *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx < m) {
    cuComplex temp = make_cuComplex(0.0f, 0.0f);
    for (int i = 0; i < WARP_SIZE; i++) {
      temp = cuCaddf(temp, phase1[i * m + idx]);
    }
    y[idx] = cuCaddf(y[idx], temp);
  }
}

__global__ void cncblasCgbmvKernelT_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                           cuComplex *phase1) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int col = idx;
  int offset = row - ku;

  if (idx < nColsA) {
    cuComplex temp = make_cuComplex(0.0f, 0.0f);
    while (row < nRowsA) {
      int i = col + offset;
      if (i >= 0 && i < m) {
        temp = cuCaddf(temp, cuCmulf(A[row * nColsA + col], x[i]));
      }
      row += gridDim.y;
      offset += gridDim.y;
    }

    phase1[blockIdx.y * nColsA + idx] = cuCmulf(*alpha, temp);
  }
}

__global__ void cncblasCgbmvKernelT_phase2(int nColsA, const cuComplex *phase1, cuComplex *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx < nColsA) {
    cuComplex temp = make_cuComplex(0.0f, 0.0f);
    for (int i = 0; i < WARP_SIZE; i++) {
      temp = cuCaddf(temp, phase1[i * nColsA + idx]);
    }
    y[idx] = cuCaddf(y[idx], temp);
  }
}

__global__ void cncblasCgbmvKernelC_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuComplex *alpha, const cuComplex *A, const cuComplex *x,
                                           cuComplex *phase1) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int col = idx;
  int offset = row - ku;

  if (idx < nColsA) {
    cuComplex temp = make_cuComplex(0.0f, 0.0f);
    while (row < nRowsA) {
      int i = col + offset;
      if (i >= 0 && i < m) {
        temp = cuCaddf(temp, cuCmulf(cuConjf(A[row * nColsA + col]), x[i]));
      }
      row += gridDim.y;
      offset += gridDim.y;
    }

    phase1[blockIdx.y * nColsA + idx] = cuCmulf(*alpha, temp);
  }
}

__global__ void cncblasCgbmvKernelC_phase2(int nColsA, const cuComplex *phase1, cuComplex *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx < nColsA) {
    cuComplex temp = make_cuComplex(0.0f, 0.0f);
    for (int i = 0; i < WARP_SIZE; i++) {
      temp = cuCaddf(temp, phase1[i * nColsA + idx]);
    }
    y[idx] = cuCaddf(y[idx], temp);
  }
}

__global__ void cncblasZgbmvKernelN_phase1(int m, int ku, int nColsA, int nRowsA,
                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                           const cuDoubleComplex *x,
                                           cuDoubleComplex *phase1) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int offset = ku - row;
  int col = idx + offset;

  if (tid < m) {
    cuDoubleComplex temp = make_cuDoubleComplex(0.0, 0.0);
    while (row < nRowsA && col >= 0) {
      if (col < nColsA && col >= 0) {
        temp = cuCadd(temp, cuCmul(A[row * nColsA + col], x[col]));
      }
      row += gridDim.y;
      col -= gridDim.y;
    }

    phase1[blockIdx.y * m + idx] = cuCmul(*alpha, temp);
  }
}

__global__ void cncblasZgbmvKernelN_phase2(int m, const cuDoubleComplex *phase1, cuDoubleComplex *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx < m) {
    cuDoubleComplex temp = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < WARP_SIZE; i++) {
      temp = cuCadd(temp, phase1[i * m + idx]);
    }
    y[idx] = cuCadd(y[idx], temp);
  }
}

__global__ void cncblasZgbmvKernelT_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                           const cuDoubleComplex *x,
                                           cuDoubleComplex *phase1) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int col = idx;
  int offset = row - ku;

  if (idx < nColsA) {
    cuDoubleComplex temp = make_cuDoubleComplex(0.0, 0.0);
    while (row < nRowsA) {
      int i = col + offset;
      if (i >= 0 && i < m) {
        temp = cuCadd(temp, cuCmul(A[row * nColsA + col], x[i]));
      }
      row += gridDim.y;
      offset += gridDim.y;
    }

    phase1[blockIdx.y * nColsA + idx] = cuCmul(*alpha, temp);
  }
}

__global__ void cncblasZgbmvKernelT_phase2(int nColsA, const cuDoubleComplex *phase1, cuDoubleComplex *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx < nColsA) {
    cuDoubleComplex temp = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < WARP_SIZE; i++) {
      temp = cuCadd(temp, phase1[i * nColsA + idx]);
    }
    y[idx] = cuCadd(y[idx], temp);
  }
}

__global__ void cncblasZgbmvKernelC_phase1(int m, int ku, int nRowsA, int nColsA,
                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                           const cuDoubleComplex *x,
                                           cuDoubleComplex *phase1) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int row = blockIdx.y;
  int col = idx;
  int offset = row - ku;

  if (idx < nColsA) {
    cuDoubleComplex temp = make_cuDoubleComplex(0.0, 0.0);
    while (row < nRowsA) {
      int i = col + offset;
      if (i >= 0 && i < m) {
        temp = cuCadd(temp, cuCmul(cuConj(A[row * nColsA + col]), x[i]));
      }
      row += gridDim.y;
      offset += gridDim.y;
    }

    phase1[blockIdx.y * nColsA + idx] = cuCmul(*alpha, temp);
  }
}

__global__ void cncblasZgbmvKernelC_phase2(int nColsA, const cuDoubleComplex *phase1, cuDoubleComplex *y) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  if (idx < nColsA) {
    cuDoubleComplex temp = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < WARP_SIZE; i++) {
      temp = cuCadd(temp, phase1[i * nColsA + idx]);
    }
    y[idx] = cuCadd(y[idx], temp);
  }
}
