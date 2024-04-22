#include "library.cuh"
#include <stdio.h>

__global__ void hello() {
  printf("Hello, world!\n");
}
