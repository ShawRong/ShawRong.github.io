---
title: "Scaling Laws"
date: 2025-07-11T04:16:22.730Z
draft: false
tags: []
---

# Neural Network Scaling Laws - Study Notes

## Core Scaling Law Formula

**L(X) = (X/X_c)^(-α_X)**

Where:

- **L** = Loss (performance metric, lower = better)
- **X** = Scale factor (D for data, N for parameters, C for compute)
- **X_c** = Critical threshold (minimum scale where power laws apply)
- **α_X** = Scaling exponent (determines improvement rate)

## Three Key Scaling Dimensions

### 1. Data Scaling: L(D) = (D/D_c)^(-α_D)

- **D** = Dataset size (training tokens/examples)
- **D_c** = Critical dataset size threshold
- **α_D ≈ 0.095** for transformers
- **Doubling data** → Loss × 2^(-0.095) ≈ **6.8% improvement**

### 2. Parameter Scaling: L(N) = (N/N_c)^(-α_N)

- **N** = Number of model parameters
- **N_c** = Critical parameter count threshold
- **α_N ≈ 0.076** for transformers

### 3. Compute Scaling: L(C) = (C/C_c)^(-α_C)

- **C** = Total compute (FLOPs)
- **C_c** = Critical compute threshold
- **α_C** varies by compute allocation

## Chinchilla Optimal Scaling

**N_opt ∝ D_opt** (approximately 1:1 ratio)

For compute budget C:

- **N_opt ∝ C^0.5** (optimal parameters)
- **D_opt ∝ C^0.5** (optimal training tokens)

## Key Insights

### Diminishing Returns

- Small exponents (α < 0.1) mean **significant scaling needed for major improvements**
- 10x increase in scale → ~1.25x improvement in loss

### Critical Thresholds

- **X_c exists** because very small scales don't follow power laws
- Below threshold = too much noise, above threshold = predictable scaling

### Trade-offs

- **Data scaling**: Modest gains, doubles training time
- **Parameter scaling**: Better gains, increases inference cost
- **Optimal allocation**: Balance parameters and data equally

## Practical Example

**Doubling Training Data:**

- Loss improvement: 2^(0.095) ≈ 1.068 (6.8% better)
- Training time: 2x longer
- Inference time: Unchanged

## Why Scaling Laws Matter

1. **Predictable performance** across orders of magnitude
2. **Resource allocation** guidance (don't overtrain small models)
3. **ROI planning** for compute investments
4. **Architecture comparison** via scaling exponents