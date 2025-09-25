---
title: "Cuda Toolkit Reading Note1"
date: 2025-09-25T09:49:22.716Z
draft: false
tags: []
---

Cuda C++ Programming guide
========

Scalable Progamming Model
--------
There are a crucial problem when we are facing multi-core system. It's hard to pragram. To solve this problem, cuda devised a programming concept, a abstraction called block. Block are logical abstractions that get mapped to physical hardware automatically without mental burgen when programming. Blocks are scheduled on SMs, i.e. Stream Multiprocessors. Each SM can run multiple blocks, and blocks can be assigned to any available SM.

Thread communication can happend only within blocks. Thread in the same block can share memory and synchronize. This is a demand comes from the design to make parallel programming more manageable. This restriction is what enables the automatic scalability, since blocks are independent, the runtime can distribute them freely.

In conclusion, by restricting communication to within blocks, Cuda forces you to write code that's naturally scalable. If blocks could talk to each other, the runtime couldn't freely move them around different processors.

Hardware Implmentation
--------
This is about the whole architecture about the GPU. The general concept comes from the Streaming Multiprocessor. Cuda program on the CPU invoke a kernel grid on the GPU, and the blocks of the grid are distributed to SMs with available execution capacity. Then the thread of these block execute concurrently on each multiprocessor, and each multiple thread blocks can execute concurrently on each SM.

These threads of the same block executes according to the SIMT Architecture. And the instructions are pipelined, too. This leverages the instruction level parallelism within a single thread, as well as extensive thread-level parallelism. There are no such branch prediction or speculative execution.




Programming Model
--------
There is a limit to the number of the threads per block, due to the constraint that all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of the core. And on current GPUs, a thread block may contain up to 1024 threads.

But, for a kernel, which can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the number of threads per block times the number of blocks.

So about block, it get its index and dim. And this works for thread, too.
So we can get a code just like this.

```c++
//Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N)
		C[i][j] = A[i][j] + B[i][j];
}

int main() {
	//Kernel invocation
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y)
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
```

There the block size is set to be 16 x 16(256 threads), is a common choice for the thread block size.

Again, the thread blocks are required to execute independently, and it allows thread blocks to be scheduled in any order and across any number of cores.

Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses(like coalesced). 

The things we can do within a block:
- \_\_syncthreads: act more like a barrier in a range of block
- Shared memory
- Some other cooperative groups

**Cooperative groups**:
In the past time before CUDA 9, CUDA programming model has provided a single simple construct for synchronizing cooperating threads: a barrier across all threads of a **thread block**, as implemented with the **\_\_syncthreads** intrinsic function. And many developers seek to a operation at other granularities, like threads in the same warp, or across sets of thread blocks running on a single GPU. 
Therefore, cooperative groups are here to provide these kinds of granularity for flexity.

Keywords: **granularity at block-level**, **granularity at warp-level** and **granularity at single GPU-level**.


New LaunchAPI
--------
This Cooperative groups also provide us some new launch APIs that enforce certain restrictions and therefore can guarantee the synchronization will work. 

Here the "launch APIs" refers to the API that start kernel from the host. Traditional CUDA launch uses the <<<>>> syntax. But now we can use the new launch APIs that provide additional guarantees and restrictions. 

```c++
cudaLaunchCooperativeKernel()
```

Restrictions:
- All thread blocks must be launched simultaneously. (Not scheduled over time that due to resource restriction, some can't launch at the first time)
- The GPU must have enough resources to run all blocks concurrently
- Certain hardware capabilities must be present

This kind of enforcement guarantees that global synchronization will work.

Some Cooperative Parallelism
--------
We can use **this_grid** api to get the object that represent all the thread in the grid, including all the thread across the grid. 

```c++
auto grid = cooperative_groups::this_grid()
//or
cooperative_groups::grid_group grid = cooperative_groups::this_grid();
```
We can use some method of this object, like:
```c++
// Wait for all threads across all blocks to reach this point
grid.sync()
```

**Producer-Consumer Parallelism**
```c++
// Simple example
__global__ void producer_consumer_kernel() {
	auto grid = cooperative_groups::this_grid();
	
	if (blockIdx.x < num_producer_blocks) {
	// select some of the block to be producer
		produce_data();
		grid_sync();
	} else {
		grid.sync() // wait for producers, this make sure all the thread should reach to the grid_sync point
		consume_data();
	}
}
```

**Opportunistic Parallelism**
```c++
__global__ void opportunistic_kernel() {
	auto grid = cooperative_groups::this_grid();
	
	while(work_available()) {
		// find work
		int work_id = atomicAdd(&global_work_counter, 1);
		if (work_id < total_work) {
			process_work_item(work_id);
		}
		grid.sync();
	}
}

```
**Note**: we can see there are many warp divergence and some mutex like atom add here. So, as a result, opportunistic parallelism is not recommended for GPUs.
**Global Synchronization**
```c++
__global__ void global_sync_kernel() {
	auto grid =  cooperative_groups::this_grid();
	phase1_computation();
	
	//sync all the thread across all blocks
	grid.sync();
	
	phase2_computation();
}
```

Before cooperative groups, you could only synchronize threads within a single block using \_\_synchtreads. But now we can synchronize across the entire grid.



Thread Block Clusters
--------
After the NVIDIA Compute Capacity 9.0, the CUDA programming model introduces an optional level of hierarchy called **Thread Block Clusters** that between Thread Blocks and Grid.


Thread blocks in a cluster are guaranteed to be co-scheduled on a GPU Processing Cluster in the GPU. The number of thread blocks in a cluster can be user-definend, and the maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. And some GPU hardware or MIG configuration which are too small to support 8 multiprocessors. So there won't be 8 multiprocessor.

This number can be queried using the cudaOccupancyMaxPotentialClusterSize API.

From the hardware perspective, there are some features to support the Thread Block Cluters. 
- Streaming Multiprocessor Coordination
- Shared Memory Architecture. The hardware provides a new distributed shared memory model where thread blocks within a cluster can access each other's shared memory spaces, creating a larger, virtually unified shared memory pool
- Hardware Synchronization. New synchronization primitives are implemented in hardware, including **cluster-wide barriers** that synchronize **all threads across multiple SMs** within the cluster.

We can enable a thread block cluster in a kernel definition or evoke a kernel using **compile-time kernel attribute** using \_\_cluster_dims\_\_(X, Y, Z) or using the CUDA kernel launch API **cudaLaunchKernelEx**.

**kernel attribute**
The cluster size using kernel attribute is fixed at compile time and then the kernel can be launched using the classical <<<, >>>. And in this way, the cluster size cannot be modified when launching the kernel

```c++
// Kernel Definition
// cluster size in 2, 1, 1
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel (float* input, float* output) {

}

int main() {
	float* input, *output;
	//kernel invocation	
	dim3 threadsPerBlock(16, 16);
	dim3 numBlock(N / threadsPerBlock.x, N / threadsPerBlock.y);

	//
	cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}

```
Here we can see the grid size is still counted according to block. But we should be aware of the grid size should be multiple of the cluster size. The grid size must be divisible by the cluster size.

If we get a cluster to be in size of 3. The grid size would be invalid if it's 1, 2, 4, 5, ..., etc. Anything not divisible by 3.

The thing is, even though blocks are grouped into cluster of 3, we still need to specify the total number of blocks when launching the kernel

**launch API**

```c++
// Kernel definition
// no compile time attribute there
__global__ void cluster_kernel(float *input, float *output) {

}


int main() {
	float *input, *output;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	
	// Kernel invocation with runtime cluster size
	{
		cudaLaunchConfig_t config = {0}:
		config.gridDim = numBlocks;
		config.blockDim = threadsPerBlock;
		
		cudaLaunchAttribute attribute[1];
		attribute[0].id = cudaLaunchAttributeClusterDimension;
		attribute[0].val.clusterDim.x = 2;
		attribute[0].val.clusterDim.y = 1;
		attribute[0].val.clusterDim.z = 1;
		config.attrs = attribute;
		config.numAttrs = 1;
		
		cudaLaunchKernelEx(&config, cluster_kernel, input, output);
	}
}
```


The thread blocks in the cluster are guaranteed to be co-scheduled on a single GPC and is allowed to perform hardware-supported synchronization using API of cluster group, like cluster.sync(). We get api like num_threads() and num_blocks.

Thread blocks in a cluster have access to the distributed shared memory. They can read, write and perform atomic to any address in the distributed shared memory.

We can use another kernel attribute \_\_block\_size\_\_, which allow us to launch a grid explicitly configured with the number of thread block clusters

```c++
__cluster__dims__((2, 2, 2)) __global__ void foo();

foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
```

```cpp
// first one is the block dimention
// seond is the cluster size
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

// 8x8x8 clusters, grid of cluster
foo<<<dim3(8, 8, 8)>>>();
```

Traditional CUDA(without cluster):
```cpp
__global__ void kernel();
kernel<<<num_blocks, threads_per_block>>>();
```

With clusters:
```cpp
__block_size__((threads_per_block), (cluster_dims)) __global__ void kernel();

kernel<<<num_clusters>>>(); //launch clusters, not blocks.
```

Memory Hierarchy
--------
For a single thread, it gets **per thread registers** and **local memory**.

And for Thread block, we get **per block shared memory** that threads in the block can share.

And for Thread Block Cluster, we get shared memory of all thread blocks in a cluster from **Distributed Shared Memory**.

For Grid with Clusters, we get **Global Memory**, which is shared between all GPU kernels.
![[attachments/Pasted image 20250917100020.png]]

For special additional **read-only** memory spaces that is accessible by **all threads**: the **constant and texture** memory space.

| Memory Type         | Latency (Cycles)               | Bandwidth            | Size              | Access Pattern      | Best Use Case                     |
| ------------------- | ------------------------------ | -------------------- | ----------------- | ------------------- | --------------------------------- |
| **Registers**       | 1                              | ~8 TB/s              | 32-64KB per SM    | Private per thread  | Thread-local variables            |
| **Shared Memory**   | 1-2                            | ~1.5-2 TB/s          | 48-164KB per SM   | Block-local threads | Inter-thread communication        |
| **Constant Memory** | 1-2 (cached)<br>400-600 (miss) | ~1-1.5 TB/s          | 64KB total        | Read-only, uniform  | Small lookup tables, coefficients |
| **Texture Memory**  | 400-600                        | ~800 GB/s - 1.2 TB/s | Limited by global | Read-only, spatial  | Image processing, 2D/3D data      |
| **L1 Cache**        | 400-600                        | ~1-1.5 TB/s          | 128-192KB per SM  | Automatic caching   | Global memory acceleration        |
| **L2 Cache**        | 600-800                        | ~500-800 GB/s        | 6-40MB            | Shared across SMs   | Cross-SM data sharing             |
| **Global Memory**   | 400-800                        | ~500-900 GB/s        | GB to 80GB+       | All threads         | Large datasets, main storage      |


---------------------
**Shared Memory** VS **L1 cache** 

Shared memory and L1 cache often share the **same physical SRAM** banks on modern GPUs. But they have very different **access patterns** and behaviors, which lead to latency difference. But we can see the bandwidth is the same.

| Aspect                  | Shared Memory                    | L1 Cache                             |
| ----------------------- | -------------------------------- | ------------------------------------ |
| **Access Path**         | Direct SRAM access               | Cache lookup + tag checking          |
| **Address Translation** | Simple block-relative addressing | Full virtual-to-physical translation |
| **Hardware Overhead**   | Minimal                          | Cache control logic overhead         |
| **Hit/Miss Logic**      | No cache misses (always "hits")  | Cache hit/miss determination         |
| **Memory Controller**   | Bypassed                         | Must go through memory controller    |
**The Latency Breakdown**.

*Shared Memory (1-2 cycles):*
- Thread requests address
- Direct SRAM bank access
- Data returned
*L1 Cache (400-600 cycles):*
- Thread requests global memory address
- Address translation and tag lookup
- Cache hit/miss determination
- If miss: Go to L2/global memory (hundreds of cycles)
- If hit: Still has cache controller overhead
- Data returned

**Architecture Insight**
Physical SRAM Bank (64KB example)
├── 48KB configured as Shared Memory
│   └── Direct access path (fast)
└── 16KB configured as L1 Cache  
    └── Cache controller path (slower due to overhead)

---------------------------------------
**Constant Memory**
Constant memory uses dedicated on-chip SRAM banks, usually separate from shared memory. These banks are optimized for read-only access patterns

Cache hit: 1-2 cycles latency
Cache miss: 200-400 cycles (device memory access)


**Texture Memory**
It gets advantage from: 
- spatial locality, 2D cache optimized for nearby access
- Non-coalesced access: better than global memory for scattered reads
- Hardware filtering: bilinear, trilinear interpolation
- Clamping / Wrapping: hardware boundary handling
--------

# Unified Memory Programming

Unified Memory provides:
- Single Unified Memory Pool. A single pointer value enables all processors in the CPUs and GPUs to access this memory with all of their native memory operations. It's managed by the system, and I think there should be a mechanism just like page fault, to create real memory allocation and migrate the data only if the page fault is triggered.
- **Concurrent access** to the unified memory pool from **all processor**(CPUs and GPUs) in the system. In the previous traditional memory access pattern. We have to access to the data in separate devices separately. But now, both CPUs and GPUs can be reading from and writing to the same memory pool at the same time, rather than having to take turns. Traditionally, only one processor can access a piece of data at a time. While GPU is working on data, CPU cannot access it, and vice versa. But proper synchronization is still needed for avoiding data races.

It helps with Performance:
- Data access speed may be maximized by migrating data towards processors that access it most frequently
- We can use feature **hint** to control migration
- Total system memory usage may be reduced by avoiding duplicating memory on both CPUs and GPUs

We can focus on functionality of the program at first, but worry about data-movement later in the development cycle as a performance optimization use hint.

**System-Allocated Memory**
Memory allocated on the host with system APIs: stack variables, global-/file-scope variables, malloc()/mmap(), thread locals.
**CUDA API**
cudaMallocManaged()

Not completed

----------------

# Asynchronous Operation
We can utilize asynchronous SIMT allows:
- We can initiate memory transfers without waiting for completion. Threads can do other work, while transferring happens in the background.
- We can overlapping computation and communication
- Instead of having threads sit idle waiting for slow operations, they can work on other tasks.

A typical Asynchronous operation:
```c
// Traditional synchronous approach
__shared__ float shared_data[256];
// Copy data and wait
shared_data[tid] = global_data[tid];  // All threads wait here
__syncthreads();  // Barrier - everyone waits
result = compute(shared_data[tid]);

// Asynchronous approach
__shared__ float shared_data[256];
// Start the copy operation (doesn't wait)
async_copy(&shared_data[tid], &global_data[tid]);
// Do other work while copy happens
float temp = some_other_computation();
// Wait only when we actually need the data
wait_for_copy_completion();
result = compute(shared_data[tid] + temp);
```

Apparently, we need synchronization to use the asynchronous operations. Such as:
```c
// Wrong One
// Thread 1 starts copying data asynchronously
async_copy(shared_memory, global_memory);

// Thread 2 immediately tries to use the data
float result = compute(shared_memory[tid]); // WRONG! Data might not be ready yet


// Use synchronization
// Only this thread can wait on this barrier
cuda::barrier<cuda::thread_scope_thread> personal_barrier;
async_copy(..., personal_barrier);
personal_barrier.wait(); // Only this thread waits
```

There are two kinds of synchronization object, they can be "cuda:: barrier" or "cuda:: pipleline",

They get different scope, like:
-  cuda::thread_scope::thread_scope_thread
-  cuda::thread_scope::thread_scope_block
-  cuda::thread_scope::thread_scope_device
-  cuda::thread_scope::thread_scope_system
You can use them to the synchronization object, like:
```cpp
cuda::barrier<cuda::thread_scope_thread> personal_barrier;
async_copy(..., personal_barrier);
personal_barrier.wait();
```
Or in block synchronization
```cpp
cuda::barrier<cuda::thread_scope_block> block_barrier;
if (threadIdx.x == 0) {
	async_copy(..., block_barrier);
}
block_barrier.wait();
```