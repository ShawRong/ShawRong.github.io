---
title: "Two Aspects of Pruning"
date: 2025-07-06T10:37:09.417Z
draft: false
tags: []
---

-[claude link](https://claude.ai/public/artifacts/18d1f408-412d-42aa-af07-9853577253ba)
-[claude link chat](https://claude.ai/share/063d2d3e-1e51-4bc7-b003-b88d472b102b)

# KV Cache and Pruning Strategies - Study Notes

## What is KV Cache?

### Purpose

KV cache is a memory optimization technique used in transformer models during text generation to avoid redundant computations.

### How it Works

- **Problem**: Without caching, generating each new token requires recomputing Key (K) and Value (V) matrices for all previous tokens
- **Solution**: Store K and V representations of previous tokens, only compute K and V for the new token

### Example Process

```
Generating "The cat sat on"

Step 1: Generate "cat"
- Input: "The"
- Compute K₁, V₁ for "The"
- Cache: K=[K₁], V=[V₁]

Step 2: Generate "sat" 
- Input: "The cat"
- Compute K₂, V₂ for "cat" 
- Cache: K=[K₁, K₂], V=[V₁, V₂]
- Reuse K₁, V₁ (no recomputation!)

Step 3: Generate "on"
- Input: "The cat sat"
- Compute K₃, V₃ for "sat"
- Cache: K=[K₁, K₂, K₃], V=[V₁, V₂, V₃]
```

## KV Cache Structure

### Matrix Dimensions

- **Format**: [tokens × channels]
- **Tokens**: Sequence positions (words/subwords in the input)
- **Channels**: Feature dimensions (hidden size of the model, e.g., 768, 1024, 4096)
- **Growth**: Cache grows as sequence lengthens: [1×channels] → [2×channels] → [3×channels]...

### Key Properties

- Both K and V caches have identical dimensions
- Channels size is determined by model architecture
- Each element represents the intersection of a token and a channel

## Pruning Strategies

### Core Concepts

- **Pruning Direction**: Which axis to remove elements from
- **Output-Awareness**: Using scoring metrics to estimate element importance
- **Local Dense Window**: Keep recent 32 tokens untouched during decoding

### 1. Per-Channel Pruning

**Definition**: For each channel (column), selectively remove some token entries

**How it works**:

- Look at each channel across all tokens
- Apply different sparsity patterns to different channels
- Remove elements within each channel vector

**Example**:

```
Original:
        Ch1  Ch2  Ch3  Ch4  Ch5  Ch6
Token1: [a,   b,   c,   d,   e,   f]
Token2: [g,   h,   i,   j,   k,   l]
Token3: [m,   n,   o,   p,   q,   r]
Token4: [s,   t,   u,   v,   w,   x]

After Per-Channel Pruning:
        Ch1  Ch2  Ch3  Ch4  Ch5  Ch6
Token1: [a,   -,   c,   d,   -,   f]
Token2: [g,   h,   -,   -,   k,   l]
Token3: [-,   n,   o,   p,   q,   -]
Token4: [s,   -,   u,   v,   -,   x]
```

### 2. Per-Token Pruning

**Definition**: For each token (row), selectively remove some channel entries

**How it works**:

- Look at each token across all channels
- Apply different sparsity patterns to different tokens
- Remove elements within each token's representation

**Example**:

```
Original:
        Ch1  Ch2  Ch3  Ch4  Ch5  Ch6
Token1: [a,   b,   c,   d,   e,   f]
Token2: [g,   h,   i,   j,   k,   l]
Token3: [m,   n,   o,   p,   q,   r]
Token4: [s,   t,   u,   v,   w,   x]

After Per-Token Pruning:
        Ch1  Ch2  Ch3  Ch4  Ch5  Ch6
Token1: [a,   b,   -,   -,   e,   f]  ← 66% kept
Token2: [g,   -,   i,   j,   -,   l]  ← 66% kept
Token3: [m,   n,   -,   p,   q,   r]  ← 83% kept
Token4: [-,   t,   u,   -,   w,   x]  ← 66% kept
```

## Key Differences Between Pruning Strategies

| Aspect               | Per-Channel Pruning           | Per-Token Pruning             |
| -------------------- | ----------------------------- | ----------------------------- |
| **Direction**        | Vertical (across tokens)      | Horizontal (across channels)  |
| **Unit**             | Channel vector                | Token vector                  |
| **Sparsity Pattern** | Different for each channel    | Different for each token      |
| **What's Removed**   | Token entries within channels | Channel entries within tokens |

## Important Notes

- Both strategies create **unstructured sparsity** (irregular patterns)
- Each channel captures different features/aspects of the representation
- Each token has its own unique representation across channels
- The choice between strategies depends on the specific use case and model characteristics
- Recent tokens (last 32) are typically preserved for accuracy