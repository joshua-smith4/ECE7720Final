
__device__ void fill_with(
  float *res,
  float *gradsign,
  int len,
  int num_fill
)
{
  int resIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int gradIdx = resIdx % len;
  if (resIdx < num_fill * len)
  {
    res[resIdx] = gradsign[gradIdx];
  }
}

__device__ void mult_vec_seg(
  float *res,
  float *epsilon,
  int len,
  int num_fill
)
{
  int resIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int epsIdx = resIdx / len;
  if (resIdx < num_fill * len)
  {
    res[resIdx] *= epsilon[epsIdx];
  }
}

__device__ void add_vec_seg(
  float *res,
  float *x,
  int len,
  int num_fill
)
{
  int resIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int xIdx = resIdx % len;
  if (resIdx < num_fill * len)
  {
    res[resIdx] += x[xIdx];
  }
}

__global__ void gen_examples_fgsm(
  float *res,
  float *x,
  float *gradsign,
  float *epsilon,
  int num_examples,
  int len_example
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0)
  {
    int res_len = len_example*num_examples;
    fill_with<<<res_len, 1>>>(
      res, gradsign, len_example, num_examples);
    cudaDeviceSynchronize();
    mult_vec_seg<<<res_len, 1>>>(
      res, epsilon, len_example, num_examples);
    cudaDeviceSynchronize();
    add_vec_seg<<<res_len, 1>>>(
      res, x, len_example, num_examples);
  }
}
