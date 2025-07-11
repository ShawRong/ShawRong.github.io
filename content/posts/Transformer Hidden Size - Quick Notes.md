---
title: "Transformer Hidden Size - Quick Notes"
date: 2025-07-11T04:16:29.715Z
draft: false
tags: []
---

# Transformer Hidden Size - Quick Notes

## Core Concepts

**Hidden State**: The vector representation of each token at each layer

- Each token position has its own hidden state vector
- Content evolves through layers, but size stays constant

**Hidden Size**: The dimensionality of these vectors (e.g., 512, 768, 1024)

- Key architectural parameter
- Determines model width

## Size Relationships

### Single-Head Attention

- Q, K, V dimensions = hidden_size
- Linear projections: hidden_size → hidden_size

### Multi-Head Attention

- `d_k = d_v = hidden_size / num_heads`
- Each head: hidden_size → d_k
- After concat: back to hidden_size

**Example**: hidden_size=256, heads=4

- Per head: 256/4 = 64
- Q, K, V per head: 64 dimensions
- Concatenated: 4 × 64 = 256

## Important Rules

1. **Hidden size is constant** across all transformer layers
2. **FFN temporarily expands** (usually 4×) then contracts back
3. **Hidden size determines Q/K/V dimensions**, not vice versa
4. **Token positions** are sequence indices (0, 1, 2, ...)

## Flow Example

```
Input: [batch, seq_len, hidden_size]
Layer 1: [batch, seq_len, hidden_size] 
Layer 2: [batch, seq_len, hidden_size]
...
Output: [batch, seq_len, hidden_size]
```

Size never changes, only content evolves!