#ifndef CNCBLAS_HELPERS_CUH
#define CNCBLAS_HELPERS_CUH

#include <cuComplex.h>
#include <iostream>
#include <stdexcept>

/* ------------------------- AMAX/AMIN ------------------------- */


__device__ static __inline__
float cncblasCmag(const cuComplex *x) {
  float a = x->x;
  float b = x->y;
  float mag = fabsf(a) + fabsf(b);
  return mag;
}

__device__ static __inline__
double cncblasZmag(const cuDoubleComplex *x) {
  double a = x->x;
  double b = x->y;
  double mag = fabs(a) + fabs(b);
  return mag;
}

__device__ static __inline__
double infty() {
  const unsigned long long ieee754_inf = 0x7ff0000000000000ULL;
  return __longlong_as_double(ieee754_inf);
}

/* ------------------------- DOT ------------------------- */

__device__ static __inline__
void cncblasCVaddf(volatile cuComplex *a, volatile cuComplex *b) {
  a->x += b->x;
  a->y += b->y;
}

__device__ static __inline__
void cncblasZVadd(volatile cuDoubleComplex *a, volatile cuDoubleComplex *b) {
  a->x += b->x;
  a->y += b->y;
}

/* ------------------------- ROT ------------------------- */

template<typename T>
inline static void
rotParamErrorCheck(size_t n, T *x, T *y, const T *c, const T *s) {
  try {
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
      throw std::invalid_argument("x, y, c, and s must not be null");
    }
    const T epsilon = 1e-5;
    if (std::abs((*c) * (*c) + (*s) * (*s) - 1) > epsilon) {
      throw std::invalid_argument("c^2 + s^2 must be equal to 1");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

/* -------------------- ROTG --------------------- */

#ifndef sgnf
#define sgnf(x) ((x) > 0.0f ? 1.0f : ((x) < 0.0f ? -1.0f : 0.0f))
#endif // sgnf(x)

#ifndef sgn
#define sgn(x) ((x) > 0.0 ? 1.0 : ((x) < 0.0 ? -1.0 : 0.0))
#endif // sgn(x)

template<typename T>
inline static void
rotgParamErrorCheck(T *a, T *b, T *c, T *s) {
  try {
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
      throw std::invalid_argument("a, b, c, and s must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

template<typename T>
inline static void
rotgScalarPointerPostprocess(T *alpha, T *h_alpha) {
  if (cncblasGetMemoryType(alpha) == cudaMemoryTypeHost) {
    *alpha = *h_alpha;
  } else if (cncblasGetMemoryType(alpha) == cudaMemoryTypeDevice) {
    checkCudaErrors(cudaMemcpy(alpha, h_alpha, sizeof(T), cudaMemcpyHostToDevice));
  } else if (cncblasGetMemoryType(alpha) == cudaMemoryTypeUnregistered) {
    *alpha = *h_alpha;
  } else if (cncblasGetMemoryType(alpha) == cudaMemoryTypeManaged) {
    *alpha = *h_alpha;
  }
}

/* ------------------------- SCAL ------------------------- */

template<typename T>
inline static void
scalParamErrorCheck(size_t n, const T *alpha, T *x) {
  try {
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
    if (alpha == nullptr || x == nullptr) {
      throw std::invalid_argument("alpha and x must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

inline static void
scalParamErrorCheck(size_t n, const float *alpha, cuComplex *x) {
  try {
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
    if (alpha == nullptr || x == nullptr) {
      throw std::invalid_argument("alpha and x must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

inline static void
scalParamErrorCheck(size_t n, const double *alpha, cuDoubleComplex *x) {
  try {
    if (n <= 0) {
      throw std::invalid_argument("n must be greater than 0");
    }
    if (alpha == nullptr || x == nullptr) {
      throw std::invalid_argument("alpha and x must not be null");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

/* ------------------------- GBMV ------------------------- */

template<typename T>
inline static void
gbmvParamErrorCheck(int m, int n, int kl, int ku,
                    const T *&alpha, const T *&A, const T *&x,
                    const T *&beta, T *&y) {
  try {
    if (m < 0 || n < 0 || kl < 0 || ku < 0) {
      throw std::invalid_argument("m, n, kl, or ku is less than 0");
    }
    if (alpha == nullptr || beta == nullptr || A == nullptr || x == nullptr || y == nullptr) {
      throw std::invalid_argument("One or more input arrays are nullptr");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

/* ------------------------- GEMV ------------------------- */

template<typename T>
inline static void
gemvParamErrorCheck(int m, int n,
                    const T *&alpha, const T *&A, const T *&x,
                    const T *&beta, T *&y) {
  try {
    if (m < 0 || n < 0) {
      throw std::invalid_argument("m or n is less than 0");
    }
    if (alpha == nullptr || beta == nullptr || A == nullptr || x == nullptr || y == nullptr) {
      throw std::invalid_argument("One or more input arrays are nullptr");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

/* ------------------------- GER ------------------------- */

template<typename T>
inline static void
gerParamErrorCheck(int m, int n,
                   const T *&alpha, const T *&x, const T *&y, T *&A) {
  try {
    if (m < 0 || n < 0) {
      throw std::invalid_argument("m or n is less than 0");
    }
    if (alpha == nullptr || x == nullptr || y == nullptr || A == nullptr) {
      throw std::invalid_argument("One or more input arrays are nullptr");
    }
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    exit(1);
  }
}

#endif // CNCBLAS_HELPERS_CUH