#include <iostream>
#include "cnc_builtin_types.cuh"

__global__ void vecAdd(cncComplex *a, cncComplex *b, cncComplex *c, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    c[index] = a[index] + b[index];
    index += blockDim.x * gridDim.x;
  }
}

int main() {
  #define N (1 << 14)
  cncComplex *a, *b, *c;
  a = (cncComplex *) malloc(N * sizeof(cncComplex));
  b = (cncComplex *) malloc(N * sizeof(cncComplex));
  c = (cncComplex *) malloc(N * sizeof(cncComplex));

  cncComplex *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * sizeof(cncComplex));
  cudaMalloc(&d_b, N * sizeof(cncComplex));
  cudaMalloc(&d_c, N * sizeof(cncComplex));

  for (cncInt i = 0; i < N; i++) {
    a[i] = cncComplex(i, i);
    b[i] = cncComplex(i, i);
  }

  cudaMemcpy(d_a, a, N * sizeof(cncComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N * sizeof(cncComplex), cudaMemcpyHostToDevice);

  vecAdd<<<48, 1024>>>(d_a, d_b, d_c, N);

  cudaMemcpy(c, d_c, N * sizeof(cncComplex), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    if (c[i].real != 2 * i || c[i].imag != 2 * i) {
      std::cout << "Error at index " << i << ": " << c[i].real << " + " << c[i].imag << "i" << std::endl;
      return 1;
    }
  }
  std::cout << "Success!" << std::endl;

  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
