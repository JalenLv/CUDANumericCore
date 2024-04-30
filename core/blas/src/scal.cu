#include <level_one.cuh>
#include <iostream>
#include <stdexcept>

/* -------------------- SCAL -------------------- */

const size_t BLOCK_SIZE = 256;

__global__ void cnblasSscalKernel(size_t n, const float *alpha, float *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= *alpha;
  }
}

void cncblasSscal(size_t n, const float *alpha, float *x) {
  // Check for invalid inputs
  try {
    if (n <= 0) {
      throw std::invalid_argument("cncblasSscal: invalid n");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("cncblasSscal: alpha is null");
    }
    if (x == nullptr) {
      throw std::invalid_argument("cncblasSscal: x is null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cnblasSscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, alpha, x);
}

__global__ void cnblasDscalKernel(size_t n, const double *alpha, double *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= *alpha;
  }
}

void cncblasDscal(size_t n, const double *alpha, double *x) {
  // Check for invalid inputs
  try {
    if (n <= 0) {
      throw std::invalid_argument("cncblasDscal: invalid n");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("cncblasDscal: alpha is null");
    }
    if (x == nullptr) {
      throw std::invalid_argument("cncblasDscal: x is null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cnblasDscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, alpha, x);
}

__global__ void cnblasCscalKernel(size_t n, const cuComplex *alpha, cuComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmulf(x[i], *alpha);
  }
}

void cncblasCscal(size_t n, const cuComplex *alpha, cuComplex *x) {
  // Check for invalid inputs
  try {
    if (n <= 0) {
      throw std::invalid_argument("cncblasCscal: invalid n");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("cncblasCscal: alpha is null");
    }
    if (x == nullptr) {
      throw std::invalid_argument("cncblasCscal: x is null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cnblasCscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, alpha, x);
}

__global__ void cnblasCsscalKernel(size_t n, const float *alpha, cuComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmulf(x[i], make_cuComplex(*alpha, 0.0f));
  }
}

void cncblasCsscal(size_t n, const float *alpha, cuComplex *x) {
  // Check for invalid inputs
  try {
    if (n <= 0) {
      throw std::invalid_argument("cncblasCsscal: invalid n");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("cncblasCsscal: alpha is null");
    }
    if (x == nullptr) {
      throw std::invalid_argument("cncblasCsscal: x is null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cnblasCsscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, alpha, x);
}

__global__ void cnblasZscalKernel(size_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmul(x[i], *alpha);
  }
}

void cncblasZscal(size_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x) {
  // Check for invalid inputs
  try {
    if (n <= 0) {
      throw std::invalid_argument("cncblasZscal: invalid n");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("cncblasZscal: alpha is null");
    }
    if (x == nullptr) {
      throw std::invalid_argument("cncblasZscal: x is null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cnblasZscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, alpha, x);
}

__global__ void cnblasZdscalKernel(size_t n, const double *alpha, cuDoubleComplex *x) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = cuCmul(x[i], make_cuDoubleComplex(*alpha, 0.0));
  }
}

void cncblasZdscal(size_t n, const double *alpha, cuDoubleComplex *x) {
  // Check for invalid inputs
  try {
    if (n <= 0) {
      throw std::invalid_argument("cncblasZdscal: invalid n");
    }
    if (alpha == nullptr) {
      throw std::invalid_argument("cncblasZdscal: alpha is null");
    }
    if (x == nullptr) {
      throw std::invalid_argument("cncblasZdscal: x is null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  // Launch the kernel
  size_t GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cnblasZdscalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, alpha, x);
}
