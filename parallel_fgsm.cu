#include <cublas_v2.h>

__global__ void doublify(float *a)
{
  int idx = threadIdx.x + threadIdx.y*4;
  a[idx] *= 2;
}
