#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cncblas.h>

const int N = 1 << 10;

TEST(axpy, singlePrecision) {
  float *alpha;
  float *h_x, *h_y;
  float *d_x, *d_y_cnc, *d_y_cublas;

  h_x = new float[N];
  h_y = new float[N];
}
