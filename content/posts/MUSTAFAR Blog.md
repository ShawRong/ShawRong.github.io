---
title: "MUSTAFAR Blog"
date: 2025-07-11T02:42:22.527Z
draft: false
tags: []
---

# Key statement from Abstract
- they claims **unstructured sparsity** enables sparsity levels up to **70%** **without compromising accuracy** or requiring fine-tuning.
- they did exploration of pruning strategies, and find **per-token magnitude-based pruning** as highly effective for both Key and Value caches under **unstructured sparsity**, surpassing prior structured pruning schemes.
- Value cache surprisingly benefits from a simple **magnitude-based pruning** despite its uniform distribution.
- They use **a bitmap-based sparse format** and **a custom attention kernel** capable of compressing and directly **computing over** caches pruned to arbitrary sparsity patterns.
- Their kernel coupled with the bitmap-based format delivers substantial compression of KV cache up to **45%** of dense inference and increased tokens/sec throughput of up to **2.23x** compared to dense inference.

# Note of Introduction
Effective pruning of the KV cache entails two core challenges:
- achieving substantial **reduction in KV cache** size while preserving model accuracy
- ensuring that the runtime pruning and compression processes are sufficiently efficient. (the associated overhead **must not outweigh** the latency gains introduced by the resulting sparsity)

# Note of Pruning Algorithm for Unstructured Sparsity
They key 2 concepts for kv cache pruning: 
- Pruning Direction: the axis along which sparsity is induced. (column-wise or row-wise).
- Output Awareness: scoring metric that serves as a proxy for estimating each element's contribution to the operation's output.

They employed **local dense window** in the exploration, where the recent 32 tokens remain untouched during the decode phase.


# Note of Pruning Key Cache
About the key cache, the author cited the observation of KIVI, that key cache exhibits distinct channel-wise outlier. 

The author compared three scoring method, including ThinK, per-token magnitude-based unstructured pruning and output-aware unstructured pruning.

The result: per-token unstructured pruning achieve better result, and output-aware mechanism can slightly improve.

So the author draw a conclusion: while "outlier channels" show promise for per-channel structured pruning, unstructured sparsity achieves higher accuracy at greater sparsity levels--even without output-awareness.



# Note of Pruning Value Cache
There is a observation that value cache exhibits a more uniform distribution of activations. Making it challenging to apply the same channel-wise pruning without substantial degradation in model accuracy.

And the author draw a conclusion that for value cache pruning, per-token magnitude-based pruning is already output-aware. (small value can lead to small output, it's not very rigorous conclusion)

The author compared method: ThinK(structured), magnitude(per-channel), output-aware(per-channel) and magnitude(per-token), and draw a conclusion:

unstructured pruning methods(channel or per-token) outperform structured pruning(ThinK). token-wise pruning best preserves model accuracy even at high sparsity levels. while channel-wise pruning can achieve comparable accuracy with output-awareness, token-wise pruning offers advantages in both efficiency and modularity.

# Note of Sparse Attention Kernel

The matrix-vector products(MVs) is a memory-bound operation on GPUs, which is widely used in attention mechanism.

To exploit this property, bitmap-based sparse format is adopted to minimize I/O time. 

The compression is implemented using Triton.

There are two types of attention computation in the decode stage. SpMV for the compressed KV cache, and dense MV for the KV cache with the local window(dense).

About the SpMV, the kernel developed by the author follows the load-as-compressed, compute-as-dense paradigm. 
In detail, the compressed KV cache is loaded from GPU global memory into registers in compressed form. 
And it is decompressed into shared memory, and then used for tile-wise dense computation.