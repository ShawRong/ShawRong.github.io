---
title: "Cuda Memo"
date: 2025-05-28T09:06:17.777Z
draft: false
tags: []
---

# Hello world
```c
#include <stdio.h>
#include <cuda.h>

/*Device code: runs on GPU*/
__global__ void Hello(void) { // always return type with void
	printf("Hello from thread %d!\n", threadIdx.x);
}

/*Host code: runs on CPU*/
int main(int argc, char* argv[]) {
	int thread_count;
	thread_count = strtol(argv[1], NULL, 10);
	Hello <<<1, thread_count>>>(); // 1st arg specify how many SMs, 2nd arg specify how many SPs in each SMs. i.e. 1st arg specify the number of thread blocks. 2nd arg specifies the number of threads in each thread block.
	cudaDeviceSynchronize();  // this will cause main program to wait until all the threads have finished.
	return 0;
}

```

```shell
nvcc -o cuda_hello cuda_hello.cu
./cuda_hello 1
```


# Build-in variables
- threadIdx: the rank or index of the thread in its thread blocks
- blockDim: the dimensions, shape, or size of the thread of blocks (how many thread in block)
- blockIdx: the rank or index of the block within the grid
- griddim: the dimensions, shape, or size of grid. (how many block in grid)
they all get fields x, y, and z

```c
int blk_ct, th_per_blk;
...
Hello <<<blk_ct, th_per_blk>>>();
//there gridDim.x = blk_ct, blockDim.x = th_per_blk

//if 3 dim,
dim3 grid_dims, block_dims;
grid_dims.x = 2;
grid_dims.y = 3;
grid_dims.z = 1;
block_dims.x = 4;
block_dims.y = 4;
block_dims.z = 4;
Kenerl <<<grid_dims, block_dims>>>();
```

# Vec-add
```c
__global__ void Vec_add(
	const float x[], //in
	const float y[], //in
	float z[],       //out
	const int n //in
) {
	int my_elt = blockDim.x * blockIdx.x + threadIdx.x ;  //important
	if (my_elt < n) // global variable n
		z[my_elt] = x[my_elt] + y[my_elt];
}
```
key idea: assign the iterations of a loop to a individual threads

# Memory Malloc
We can use normal malloc in cuda c, but only for host.
- cudaMalloc, it allocate memory on the device
- cudaMallocManaged, it allocates memory that can be accessed by both CPU and GPU (auto transfer, only supported on newer version, Unified Memory). 
- cudaHostAlloc, allocates pinned host memory for faster CPU-GPU transfers.
- cudaMemcpy, copy memo, use key word cudaMemcpyHostToDevice,etc.



# API
## cudaFree
```c
__host__ __device__ cudaError_t cudaFree(void* ptr)
//__device__ indicate this function can be called from the device
// it frees storage allocated on the device.
```
## cudaMemcpy
```c
__host__ cudaError_t cudaMemcpy(
	void* dest, //out, dest is the first.
	const void* source, //in
	size_t count, //in
	cudaMemcpyKind kind //in cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
) // this function is synchronous
```

## Returing results from cuda kernels
```c
// wrong way
__global__ void Add(int x, int y, int* sum_p) {
	*sum_p = x + y;  // invalid address in device
}
int main(void) {
	int sum = -5;
	Add <<<1, 1>>> (2, 3, &sum);
	cudaDeviceSynchronize();
	printf("The sum is %d\n", sum);
}

//using managed memory
__global__ void Add(int x, int y, int* sum_p) {
	*sum_p = x + y;  // invalid address in device
}
int main(void) {
	int* sum_p;
	cudaMallocManaged(&sum_p, sizeof(int));
	*sum_p = -5
	Add <<<1, 1>>> (2, 3, sum_p);
	cudaDeviceSynchronize();
	printf("The sum is %d\n", sum);
}

//using explicity copying 
__global__ void Add(int x, int y, int* sum_p) {
	*sum_p = x + y;  // invalid address in device
}
int main(void) {
	int* hsum_p, *dsum_p;
	hsum_p = (int*) malloc(sizeof(int));
	cudaMalloc(&dsum_p, sizeof(int));
	*hsum_p = -5
	Add <<<1, 1>>> (2, 3, dsum_p);
	cudaMemcpy(hsum_p, dsum_p, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("The sum is %d\n", sum);
	free(hsum_p);
	cudaFree(dsum_p);
}

// we can a managed qualifier, it's just like a variable managed by cudaMallocManaged
__managed__ int sum;
__global__ void Add(int x, int y) {
	sum = x + y;
}
int main(void) {
	sum = -5;
	Add <<<1, 1>>> (2, 3);
	cudaDeviceSynchronize();
	printf("After kernel: The sum is %d \n", sum);
}
```

## Recall of serial trapezoidal rule
```c
float Serial_trap(
	const float a, //in
	const float b, //in
	const float n, //in
) {
	float x, h = (b - a) / n;
	float trap = 0.5 * (f(a) + f(b));
	for (int i = 1; i <= n - 1; i++) {
		x = a + i * h;
		trap += f(x);
	}
	trap = trap * h;
	return trap;
}
```

A kernel version:
```c
__global__ void Dev_trap(
	const float a, // in
	const float b, // in
	const float h, // in
	const float n, // in
	float* trap_p //in and out
) {
	int my_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (0 < my_i && my_i < n) {
		float my_x = a + my_i * h;
		float my_trap = f(my_x);
		atomicAdd(trap_p, my_trap);
	}
}

void Trap_wrapper(
	const float a, //in
	const float b, //in
	const float n, //in
	float* trap_p, //out
	const int blk_ct, //in
	const int th_per_blk, //in
) {
	// trap p is allocated in main with cudaMallocManaged, and it's shared for all threads
	*trap_p = 0.5 * (f(a) + f(b));
	float h = (b - a) / n;
	Dev_trap<<<blk_ct, th_per_blk>>>(a, b, h, n, trap_p);
	cudaDeviceSynchronoize();
	*trap_p = h*(*trap_p);
}
```

## warp shuffle
warp shuffle allow a collection of threads within a warp to read variables stored by other threads in the warp.

In cuda, warp is a set of threads with consecutive ranks belonging to a thread block. The threads in a warp operate in SIMD fashion. So threads in different warps can execute different statements with no penalty. If threads in a warp take different branches in an if-else, the threads are said to have diverged.

The rank of a thread within a warp is called the thread's lane, and it can be computed using the formula:
```c
lane = threadIdx.x % warpSize;
```


# Coalesced Access
If memory address accessed by threads in the same thread block are consecutive, then these memory accesses are grouped into one memory transaction.


# Qualifier
# __device__
- callable from the device only
- synchronous

## host
- can be used with device to compile the function both on host and the device

# Variable Type Qualifiers
a variable declared in device code without a type qualifier typically resides in the register.
## device
declare a variable that resides on the device.
- resides in global memory space
- has lifetime of an application 
- is accessible from all the threads within the grid
```c
__device__ int d_value;
__global__ void test_Kernel()
{
	int threadID = threadIdx.x;
	d_value = 1;
	printf("threadID %-3d d_value %3d\n", threadID, d_value);
}
```
## constant
- reside in constant memory space
- is accessible from all the threads within the grid and fom the host
## shared
- resides in the shared memory space of a thread block
- has lifetime of the block
- is accessible from all the threads within the block