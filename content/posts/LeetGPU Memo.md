---
title: "LeetGPU Memo"
date: 2025-08-21T09:38:51.826Z
draft: false
tags: []
---

reduce kernel
```c
#include <cuda_runtime.h>

  

#define T 256

// we need a static number when declaring shared memory

__global__ void reduction_kernel(const float* input, float* output, int N) {
	__shared__ float shared[T];

	int tid = threadIdx.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	  
	// init the shared memory of each block
	shared[tid] = (idx < N) ? input[idx] : 0.0f;
	__syncthreads();
	
	// reduction
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}
	// inverse way
//	for (int s = 1; s < blockDim.x; s <<= 1) {
//		if (tid % (2 * s) == 0) {
//			shared[tid] += shared[tid + s];
//		}
//		__syncthreads();
//	}
	
	if (tid == 0) {
		atomicAdd(output, shared[0]);
	}
}
// input, output are device pointers

extern "C" void solve(const float* input, float* output, int N) {
	int threadsPerBlock = T;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
	cudaDeviceSynchronize();
}
```

Count array element
------------
```cpp
#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id < N) {
		if(input[thread_id] == K) {
			atomicAdd(output, 1);
		}
	}
}


extern "C" void solve(const float* input, float* output, int N) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
	cudaDeviceSynchronize();
}
```