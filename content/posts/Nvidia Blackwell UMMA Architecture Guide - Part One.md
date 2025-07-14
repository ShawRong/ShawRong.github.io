---
title: "Nvidia Blackwell UMMA Architecture Guide - Part One"
date: 2025-07-14T07:15:24.147Z
draft: false
tags: []
---

# NVIDIA Blackwell UMMA Architecture Guide - Part One

## Overview

This guide covers the fundamental concepts of NVIDIA's Blackwell GPU architecture, focusing on the transition from Hopper's WGMMA to Blackwell's UMMA (Unified Matrix Multiply-Accumulate) instruction and the introduction of Tensor Memory (TMEM).

## 1. From Hopper WGMMA to Blackwell UMMA

### WGMMA (Hopper Architecture)

- **Full Name**: Warp Group Matrix Multiply-Accumulate
- **Nature**: Asynchronous instruction for matrix operations on Tensor Cores
- **Launch Model**: Multi-threaded (multiple threads coordinate to launch)
- **Benefits of Async**: Enables overlap of computation with other work, better resource utilization

### UMMA (Blackwell Architecture)

- **Full Name**: Unified Matrix Multiply-Accumulate (CUTLASS terminology for `tcgen05.mma`)
- **Why tcgen05**: Tensor Core Generation 5 (Blackwell = 5th gen Tensor Cores)
- **Launch Model**: Single-threaded (only one thread launches the operation)
- **Operations Supported**:
    - `D = A × B + D` (multiply-accumulate)
    - `D = A × B` (multiply only)

### Key Evolution: TMA → UMMA Analogy

- **TMA (Tensor Memory Accelerator)**: Made data copying single-threaded and register-efficient
- **UMMA**: Applies the same principles to matrix operations
- Both follow the pattern: **offload complexity from software to dedicated hardware**

## 2. Tensor Memory (TMEM)

### What is TMEM?

- **Definition**: Dedicated on-chip memory for UMMA accumulation operations
- **Purpose**: Fast storage for intermediate matrix computation results
- **Capacity**: 128 rows (fixed) × variable columns

### TMEM Allocation

```cpp
// Allocation syntax
tcgen05.alloc.b32 %tmem_descriptor, num_columns;

// Requirements:
// - Minimum 32 columns
// - Must be power of 2 (32, 64, 128, 256, etc.)
// - Allocation returns a descriptor/address
// - Must explicitly deallocate with tcgen05.dealloc
```

### TMEM vs Other Memory Types

```
TMEM ≠ Shared Memory
├── TMEM: Dedicated tensor computation space
└── Shared Memory: Stores TMEM descriptors/addresses for coordination
```

### Memory Access Restrictions

- **Per-Warp Access**: Each warp can only access specific lanes
    - Warp 0: Lanes 0-31
    - Warp 1: Lanes 32-63
    - Warp 2: Lanes 64-95
    - Warp 3: Lanes 96-127
- **Implication**: TMEM cannot be used for inter-warp data exchange

## 3. UMMA Operation Details

### Matrix Operation Capabilities

- **Supported Shapes**:
    - 64 × N × 16 (N = multiple of 8, max 256)
    - 128 × N × 16 (N = multiple of 16, max 256)
- **Largest Atom**: 128 × 256 × 16 (twice the size of largest WGMMA)

### Performance Optimization

- **Pipeline Efficiency**: Largest UMMA uses only 50% of TMEM
- **Benefit**: Multiple UMMA operations can pipeline without performance loss
- **Result**: Continuous execution, maximum throughput

### Input Descriptors

- **Matrix Descriptors**: 64-bit values containing address, layout, and swizzling info
- **Special Case**: If matrix A comes from TMEM, descriptor is replaced by simple TMEM address
- **Instruction Descriptor**: 32-bit metadata containing:
    - Data type and sparsity information
    - Transpose/negate flags for A and B matrices
    - Accumulation control (`enable-input-d`)

## 4. Key Features and Capabilities

### Data Layout and Swizzling

- **Swizzling**: Data rearrangement to optimize hardware access patterns
- **Purpose**: Avoid memory bank conflicts, enable coalesced access
- **Expected Layout**: K-major format in shared memory
- **Hardware Transpose**: "Free" transpose during memory read (no computation cost)

### Advanced Features

1. **Sparsity Support**: Hardware optimization for matrices with many zeros
2. **Transpose/Negate**: Built-in matrix transformations during operation
3. **Accumulation Control**:
    - Zero out: `D = A × B` (fresh start)
    - Accumulate: `D = A × B + D` (add to existing)

### CTA Pairs and Multi-SM Coordination

- **CTA Pair**: Two adjacent CTAs within an SM cluster working together
- **Launch Model**: Even with CTA pairs, only one thread in one CTA launches UMMA
- **Hardware Coordination**: Automatic coordination between CTAs

## 5. Memory Movement Operations

### TMEM Data Flow

```
Data IN:  UMMA operations → TMEM
Data OUT: tcgen05.ld → RMEM (registers)
Manual:   tcgen05.cp (SMEM→TMEM), tcgen05.st (RMEM→TMEM)
```

### Memory Space Terminology

- **GMEM**: Global Memory
- **SMEM**: Shared Memory
- **TMEM**: Tensor Memory
- **RMEM**: Register Memory (registers)

## 6. Epilogue Processing

### Definition

**Epilogue**: Post-processing operations after main matrix multiplication

- Activation functions (ReLU, sigmoid)
- Bias addition, scaling
- Data type conversion
- Storage to global memory

### Warpgroup Requirement

- **Problem**: Large UMMA results span entire TMEM (all 128 lanes)
- **Solution**: Entire warpgroup (4 warps) needed for epilogue
- **Process**:
    1. Each warp reads its ¼ of TMEM (32 lanes)
    2. Each warp processes its portion independently
    3. Each warp stores results to global memory

## 7. Programming Model Simplification

### Before (WGMMA)

- Multi-threaded coordination required
- Complex register management across threads
- Higher software complexity

### After (UMMA)

- Single-threaded launch
- Hardware manages complexity
- Simplified programming model
- Register-efficient design

---

## Next: Part Two Preview

The next part will cover:

- 2-CTA UMMA operations
- Advanced CUTLASS utilities
- Detailed swizzling patterns
- Performance optimization strategies