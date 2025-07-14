---
title: "Black well note claude 2"
date: 2025-07-14T10:51:18.523Z
draft: false
tags: []
---

# NVIDIA Blackwell Mixed-Precision GEMM Notes

## Overview

This note covers low-precision computation in NVIDIA Blackwell architecture, focusing on mixed-precision GEMM operations with sub-byte formats (FP8, FP6, FP4) and their implementation in CUTLASS.

## Key Concepts

### TMA (Tensor Memory Accelerator)

- **Purpose**: Hardware unit for efficient memory transfers between Global Memory (GMEM) and Shared Memory (SMEM)
- **Key Features**:
    - Automated multi-dimensional tensor transfers (1D to 5D)
    - Asynchronous operation (overlaps with computation)
    - Data format transformations during transfer
    - Layout conversions, precision conversions, sub-byte unpacking
    - Scatter/gather operations, padding, boundary handling

### Mixed-Input UMMA

- **Definition**: UMMA operations where matrices A and B can have different data types
- **Example**: Matrix A (FP8) × Matrix B (FP6) → Matrix C (FP16)
- **PTX Instruction**: `tcgen05.mma.mixed.m16n8k32.kind::f8f6f4`

## Data Format Transformations

### Packed vs Unpacked Formats

#### Packed Format (Storage in GMEM)

```
FP4: [A1A2][B1B2] - 2 values per byte
FP6: [A1A2A3][B1B2B3] - 4 values per 3 bytes  
FP8: [A1][B1] - 1 value per byte
```

#### Unpacked Format (Required by f8f6f4 UMMA)

```
FP4: [A1--][A2--][B1--][B2--] - 1 value per byte (padded)
FP6: [A1--][A2--][B1--][B2--] - 1 value per byte (padded)
FP8: [A1][B1] - 1 value per byte (unchanged)
```

### TMA's Role in Unpacking

- **Input**: Packed data in GMEM
- **Process**: Automatic unpacking during transfer
- **Output**: Unpacked data in SMEM (UMMA-friendly format)
- **Key Point**: Data precision unchanged, only memory layout reorganized

## f8f6f4 UMMA Constraints

### Fixed Dimensions

- **K extent**: Always 32 elements
- **Memory requirement**: 32 elements × 1 byte = 32 bytes in SMEM
- **Reason**: Hardware constraint for mixed-precision operations

### TMA Alignment Requirements

- **Base address**: 32B aligned (vs usual 16B)
- **Leading dimension**: Multiple of 128 elements
- **Swizzling**: Only 128B patterns supported

### CUTLASS Stricter Alignment

- **FP4 data**: 64-byte aligned (128 elements × 0.5 bytes = 64 bytes)
- **FP6 data**: 96-byte aligned (128 elements × 0.75 bytes = 96 bytes)
- **Purpose**: Ensures every row's first element meets TMA alignment requirements

## Memory Source Limitations

### UMMA Operand Sources

- **Allowed**: A from TMEM, B from SMEM ✓
- **Allowed**: A from SMEM, B from SMEM ✓
- **Not Allowed**: A from TMEM, B from TMEM ❌
- **Not Allowed**: A from SMEM, B from TMEM ❌

### TMEM Requirements

- All sub-byte data must be padded to 1 byte per value
- Only operand A can source from TMEM
- Operand B restricted to SMEM only

## DeepSeek's Two-Level Accumulation

### The Problem

- FP8 Tensor Cores use ~14-bit precision accumulation (not full FP32)
- Causes training inaccuracies for large models

### DeepSeek's Solution

1. **Level 1**: 4 consecutive WGMMA operations in Tensor Cores (FP8 accumulation)
2. **Level 2**: Add partial result to FP32 accumulator using CUDA Cores
3. **Benefits**: Speed of FP8 + accuracy of FP32 accumulation

## Alternative Data Types

### mxf4 Type

- **Supports**: Packed SMEM format (2 FP4 values per byte)
- **Usage**: FP4-only operations (not mixed-precision)
- **Advantage**: Better memory efficiency
- **TMA Type**: `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`

### CuTe Integration

#### Type Transformation in CUTLASS

```cpp
// User specifies
using ElementA = cutlass::float_e2m3_t;  // Packed FP8

// Builder transforms to
using ElementAMma = cutlass::float_e2m3_unpacksmem_t;  // Unpacked FP8
```

#### SMEM Layout Selection

```cpp
// Unified layout for all sub-byte types (after unpacking)
using ElementAMma_SmemAllocType = 
    cute::conditional_t<cute::sizeof_bits_v<ElementAMma> < 8, 
                        uint8_t, ElementAMma>;

// Architecture-specific layout optimization
using SmemLayoutAtomA = 
    decltype(sm100_smem_selector<...>());  // SM 100 = Blackwell
```

## Architecture Evolution

### SM (Streaming Multiprocessor) Generations

- **SM 70**: Volta (V100)
- **SM 80**: Ampere (A100)
- **SM 90**: Hopper (H100)
- **SM 100**: Blackwell (B100, GB200)

### Blackwell-Specific Features

- Mixed-precision UMMA (f8f6f4)
- Tensor Memory (TMEM) support
- Enhanced TMA capabilities
- New swizzling patterns for optimal performance

## Key Takeaways

1. **Mixed-precision GEMM** enables different data types for A and B matrices
2. **TMA automatically unpacks** sub-byte data during GMEM→SMEM transfer
3. **f8f6f4 UMMA requires unpacked format** (1 byte per value) in SMEM
4. **Strict alignment requirements** ensure every row meets TMA constraints
5. **CUTLASS abstracts complexity** through builder system and type transformations
6. **Architecture-specific optimizations** maximize performance on each GPU generation

## Memory Efficiency Trade-offs

|Format|Memory Usage|Access Speed|Use Case|
|---|---|---|---|
|Packed SMEM|High efficiency|Complex access|FP4-only operations|
|Unpacked SMEM|2x overhead (FP4)|Fast access|Mixed-precision operations|
|TMEM|1 byte/value|Fastest|Single operand optimization|