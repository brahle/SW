#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cassert>

#include "utilities.cuh"

/*!
 * \brief Uses the reduce algorithm to find the maximal element of
 *        an array.
 */
template< typename T >
__global__ void reduceMax(const T *g_idata, T *g_odata) {
  extern __shared__ T sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  for (unsigned int s=1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] = max(sdata[index], sdata[index + s]);
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*!
 * \brief Uses a loop to find the maximal element of an array.
 */
__global__ void dumbReduceMax(const double *data, double *res, int n) {
  *res = data[0];
  for (int i = 1; i < n; ++i) {
    if (*res < data[i]) {
      *res = data[i];
    }
  }
}

/*!
 * \brief Uses dumbReduceMax (a loop) to find the maximal element.
 */
double findMaximumDumb(const double *device_data, const int n) {
  double *izlaz_d, *izlaz_h;
  double res = 0;
  cudaError_t cudaStatus = cudaMalloc(&izlaz_d, sizeof(double));

  dumbReduceMax<<< 1, 1 >>>(device_data, izlaz_d, n);
  izlaz_h = copyArrayToHost(izlaz_d, 1);

  cudaFree(izlaz_d);
  res = izlaz_h[0];
  delete [] izlaz_h;
  return res;
}

/*!
 * \brief Method used to find maximum in an array that is on device.
 * 
 * Finds maximum in a device array. If the number of elements
 * is less than FIND_MAXIMUM_PARALLEL_CUTOFF, it calls findMaximumDumb,
 * otherwise it calls reduceMax. 
 */
double findMaximum(double *device_data, const int n) {
  double *izlaz_d;
  double res = 0;
  cudaError_t cudaStatus = cudaMalloc(&izlaz_d, n*sizeof(double));

  if (n < FIND_MAXIMUM_PARALLEL_CUTOFF) {
    return findMaximumDumb(device_data, n);
  }

  int m = 0, cnt = 0;
  for (int sz = 1; sz <= n; sz <<= 1) {
    if (sz & n) {
      int blocks = std::max(1, sz/512);
      int threads = std::min(sz, 512);
      int shared_mem = threads*sizeof(double);
      reduceMax<<< blocks, threads, shared_mem >>>(device_data+cnt, izlaz_d+m);
      cnt += sz;
      m += blocks;
    }
  }
  
  res = findMaximumDumb(izlaz_d, m);
  cudaFree(izlaz_d);
  return res;
}

/*!
 * \brief Used to test if finding maximum works. Helper function.
 */
void testMaximum(const int n) {
  double *data_host = new double[n], *data_device;
  srand(time(0));
  int id = 0;

  for (int i = 0; i < n; ++i) {
    data_host[i] = rand();
    if (data_host[id] < data_host[i]) {
      id = i;
    }
  }
  
  data_device = copyArrayToDevice(data_host, n);
  double r = findMaximum(data_device, n);
  assert(abs(data_host[id]-r) < 0.5);
  delete [] data_host;
  cudaFree(data_device);
}
