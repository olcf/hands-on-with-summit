#include "hip/hip_runtime.h"
#include <stdio.h>

// error checking macro
#define hipErrorCheck(call)                                                    \
  do {                                                                         \
    hipError_t hipErr = call;                                                  \
    if (hipSuccess != hipErr) {                                                \
      printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__,                  \
             hipGetErrorString(hipErr));                                       \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

const size_t DSIZE = 16384; // matrix side dimension
const int block_size = 256; // CUDA maximum is 1024

// matrix row-sum kernel
// we will assign one block per row
__global__ void row_sums(const float *A, float *sums, size_t ds) {

  int idx = blockIdx.x; // our block index becomes our row indicator
  if (idx < ds) {
    __shared__ float sdata[block_size];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t tidx = tid;

    while (tidx < ds) { // block stride loop to load data
      sdata[tid] += A[idx * ds + tidx];
      tidx += blockDim.x;
    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      __syncthreads();
      if (tid < s) // parallel sweep reduction
        sdata[tid] += sdata[tid + s];
    }
    if (tid == 0)
      sums[idx] = sdata[0];
  }
}

// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds) {
  // create typical 1D thread index from built-in variables
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < ds) {
    float sum = 0.0f;

    // write a for loop that will cause the thread to
    // iterate down a column, keeeping a running sum,
    // and write the result to sums
    for (size_t i = 0; i < ds; i++)
      sum += A[idx + ds * i];
    sums[idx] = sum;
  }
}


// empty kernel to jump start the GPU and get correct rocprof outputs
__global__ void init_kernel(int b) {
  int a = 12;
  int c;
  c = a+b;
}

bool validate(float *data, size_t sz) {
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {
      printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i],
             (float)sz);
      return false;
    }
  return true;
}

int main() {

  // running init_kernel to initialize GPU
  hipLaunchKernelGGL(init_kernel, dim3(DSIZE), dim3(block_size), 0, 0, 23);
  // Check for errors in kernel launch (e.g. invalid execution configuration
  // paramters)
  hipErrorCheck(hipGetLastError());
  // Check for errors on the GPU after control is returned to CPU
  hipErrorCheck(hipDeviceSynchronize());

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE * DSIZE]; // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE * DSIZE; i++) // initialize matrix in host memory
    h_A[i] = 1.0f;
  hipErrorCheck(hipMalloc(
      &d_A, DSIZE * DSIZE * sizeof(float))); // allocate device space for A
  hipErrorCheck(hipMalloc(
      &d_sums,
      DSIZE * sizeof(float))); // allocate device space for vector d_sums
  // copy matrix A to device:
  hipErrorCheck(hipMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float),
                          hipMemcpyHostToDevice));
  hipLaunchKernelGGL(row_sums, dim3(DSIZE), dim3(block_size), 0, 0, d_A, d_sums,
                     DSIZE);
  // Check for errors in kernel launch (e.g. invalid execution configuration
  // paramters)
  hipErrorCheck(hipGetLastError());
  // Check for errors on the GPU after control is returned to CPU
  hipErrorCheck(hipDeviceSynchronize());
  // copy vector sums from device to host:
  hipErrorCheck(
      hipMemcpy(h_sums, d_sums, DSIZE * sizeof(float), hipMemcpyDeviceToHost));
  if (!validate(h_sums, DSIZE))
    return -1;
  printf("row sums correct!\n");
  hipErrorCheck(hipMemset(d_sums, 0, DSIZE * sizeof(float)));
  hipLaunchKernelGGL(column_sums, dim3((DSIZE + block_size - 1) / block_size),
                     dim3(block_size), 0, 0, d_A, d_sums, DSIZE);
  // Check for errors in kernel launch (e.g. invalid execution configuration
  // paramters)
  hipErrorCheck(hipGetLastError());
  // Check for errors on the GPU after control is returned to CPU
  hipErrorCheck(hipDeviceSynchronize());
  // copy vector sums from device to host:
  hipErrorCheck(
      hipMemcpy(h_sums, d_sums, DSIZE * sizeof(float), hipMemcpyDeviceToHost));
  if (!validate(h_sums, DSIZE))
    return -1;
  printf("column sums correct!\n");
  return 0;
}
