// Description: This is a test file to check the compilation of the code
// kernel version of hello world
//

#include <cstdio>

__global__ void hello() {
  printf("Hello World\n");
}

int main() {
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();

  return 0;
}