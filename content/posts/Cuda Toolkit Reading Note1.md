---
title: "Cuda Toolkit Reading Note1"
date: 2025-09-08T11:49:19.151Z
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