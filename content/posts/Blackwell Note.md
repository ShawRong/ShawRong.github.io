---
title: "Blackwell Note"
date: 2025-07-14T11:07:38.680Z
draft: false
tags: []
---

# WGMMA and UMMA
So first, this blog starts with comparing the old version of mma comes from hopper architecture. It's called wgmma, stands for warp grouped matrix multiply-accumulation.

The feature of this wgmma, is the asynchronous instruction feature for matrix operation on Tensor core. It enables overlap of computation with other work, improve the efficiency.

The new mma called umma, which stands for unified matrix multiply-accumulate. It's lib is called tcgen05, which stands for tensor core generation5. (blackwell is the 5th gen Tensor cores)

It comes with feature: 
- single threaded(only one thread launches the operation) so you need if (tid == 0){} for usage of umma. This feature allows frees the workload of the other thread when launching mma.


So in conlusion, the difference lies in: 
- made data copying single-threaded and register-efficient. umma use a single thread to handle memory transfers instead of multiple threads.
- This single thread method avoids thread synchronization overhead and potential race conditions, and ensures predictable memory access patterns. It reduce context switching costs between threads.
- It uses tensor memory for fast storage of intermediate matrix computation result. It's a dedicated on-chip memory for UMMA accumulation

# Tensor memory
tensor memory is called tmem, too.
tmem is not shared memory, it's dedicated tensor computation space.
You can use tcgen05.alloc, tcgen05.ld to manage this.

The tensor memory is consist of 128 lane(row), and 512 columns. each cell is 32-bit(4byte).

each warp can only access to specific lanes:
- warp0: lane 0-31
- warp1: lane 32-63
- ...
- warp3: lane 96-127
obviously, tmem is not used for inter-warp data exchange. 

# UMMA details of operation
to be continue.