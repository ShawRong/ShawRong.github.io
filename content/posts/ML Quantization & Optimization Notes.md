---
title: "ML Quantization & Optimization Notes"
date: 2025-07-14T11:10:19.097Z
draft: false
tags: []
---

# ML Quantization & Optimization Notes

## Core Quantization Concepts

### Microscaling

- **Definition**: Quantization technique applying different scaling factors to small groups of values (2-4 elements)
- **Benefit**: More fine-grained precision control vs. tensor-wide scaling
- **Use case**: Better handling of outliers while maintaining efficiency

### FP4 vs INT4

**FP4 (4-bit Floating Point)**:

- Structure: Sign + Exponent + Mantissa in 4 bits
- Common formats: [1|2|1] or [1|3|0] bits
- Non-uniform value spacing (denser near zero)
- Better dynamic range and outlier handling

**INT4 (4-bit Integer)**:

- Fixed-point representation with uniform spacing
- Simpler computation but prone to saturation
- Limited dynamic range

### FP4 with Microscaling Architecture

```
Traditional: [S|EE|M] per value (4 bits each)
Microscaled: [EE|EE] + [S|M|S|M|S|M|S|M]
            ^shared    ^individual mantissas
```

- **Efficiency**: ~2.5 bits/value for groups of 4
- **Implementation**: Shared exponent decoder + multiple mantissa units

## Floating Point Fundamentals

### Exponent-Mantissa Structure

```
Value = (-1)^sign × (1 + mantissa) × 2^(exponent - bias)
```

- **Exponent**: Provides dynamic range (wide magnitude coverage)
- **Mantissa**: Provides precision (significant digits)
- **Benefit**: Consistent relative precision across magnitudes

### Clipping

- **Definition**: Constraining values to [min, max] range
- **Applications**: Gradient clipping, activation clipping, quantization clipping
- **Purpose**: Prevent overflow/saturation issues

## Fisher Information & Uncertainty

### Fisher Information Matrix

**Full Matrix**:

```
F_ij = E[(∂log p(x|θ)/∂θᵢ)(∂log p(x|θ)/∂θⱼ)]
```

- Measures parameter sensitivity and information content
- Off-diagonal terms capture parameter interactions
- O(n²) computation complexity

**Diagonal Approximation**:

```
F_ii = E[(∂log p(x|θ)/∂θᵢ)²]
```

- Assumes parameter independence
- O(n) computation - much more tractable
- Estimated by averaging squared gradients over calibration data

### Why Squared Gradients Work

```
F_ii ≈ (1/N) Σₙ (∇log p(xₙ|θ)/∂θᵢ)²
```

- Empirical approximation of expectation over data distribution
- Higher values indicate more sensitive/informative parameters
- Used to guide quantization precision allocation

### Uncertainty Quantification

- **Purpose**: Measure confidence in model predictions
- **Method**: Parameter uncertainty → prediction uncertainty
- **Relationship**: `Var(θᵢ) ≈ (F⁻¹)ᵢᵢ`

## Calibration Set Selection

### Purpose

- Determine optimal quantization parameters (scales, zero-points, clipping thresholds)
- Representative subset for post-training quantization

### Selection Criteria

- **Size**: 100-1000 samples (accuracy vs efficiency balance)
- **Representativeness**: Must capture deployment data distribution
- **Diversity**: Cover different modes, edge cases, typical examples
- **Stratification**: Ensure all classes/categories represented

### Selection Strategies

```python
# Random sampling
calib_set = random.sample(train_set, 1000)

# Stratified sampling
samples_per_class = total_samples // num_classes
calib_set = [random.sample(class_data, samples_per_class) 
             for class_data in grouped_by_class]

# Clustering-based
features = extract_features(train_set)
clusters = kmeans(features, n_clusters=100)
calib_set = [select_representative(cluster) for cluster in clusters]
```

### Best Practices

- Use validation set (avoid test set leakage)
- Monitor activation statistics during selection
- Include domain-specific variations (lighting, vocabulary, etc.)
- Sometimes create separate "calibration split" during data prep

## Key Mathematical Notation

- **E_v[]**: Expectation with respect to distribution v
- **∇log p(x|θ)**: Gradient of log-likelihood (computed via backpropagation)
- **F_ij**: Fisher information between parameters i and j
- **θ**: Model parameters