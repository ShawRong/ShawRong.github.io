---
title: "Sparsity"
date: 2025-07-04T14:16:29.661Z
draft: false
tags: []
---

# How sparsity is employed in Acceleration
When multiplying by zero or very small values, we can avoid the computation entirely.

If 90% percent of values are zero, we can theoretically only need to compute 10% of the operations.

The hardware like GPU support this acceleration in hardware way.

It can: 
- Reduce memory bandwidth. (no load zero values)
- Fewer arithmetic operations. (skip some multiplications and addition)
- Lower energy consumption. (based on previous 2 optimization)


# Three types of Sparsity in LLM
**Weight Sparsity**: Zeros in model parameters. These weights pruned during **training or post-training**. 

**Activation Sparsity**: Zero in intermediate activations during forward pass, typically from **RELU-like** functions that output zero for negative inputs. *This sparsity is input-dependent and changes dynamically based on the data being processed.*

**Attention Sparsity**: Zeros or near-zeros in attention weight matrices. Many attention heads **focus on only a subset of tokens**, creating natural sparsity patterns. *This is also input-dependent and varies across different sequences*


# Random(Unstructured) and Structured Pattern
**Random(Unstructured) Sparsity**: zero values appear scattered throughout the tensor without particular pattern.
**Drawback**:
- Irregular memory access patterns
- Hard to vectorize operations
- Requires complex indexing schemes

**Structured Pattern**: zero values follow regular patterns like entire rows, columns, or blocks being zero (and 2:4).

**Advantages**:
- Regular memory access patterns
- Easier to map to hardware parallelism


# How Structured Pattern Come Out
**Training-Time Methods**: 
Structured **Pruning** During Training: 
- Block-wise pruning: remove entire blocks of weights
- Channel pruning: Remove entire channels/filters in convolutional layer or attention heads
- N:M sparsity: For every M consecutive weights, exactly N are forced to zero (e.g. 2:4)



**Regularization** with structure constraints: add penalty terms to the loss function that encourage structured patterns:
- Group LASSO regularization to zero out entire groups of weights
- Structured dropout that follows the desired sparsity pattern


**Post-Training Methods**:
Structured **Pruning**: Take a dense pre-trained model and apply structured pruning:
- Magnitude-based: Within each block/group, keep only the largest weights and zero the rest
- Gradient-based: Use gradient information to decide which structure to prune
- Fisher information: Use second-order information to make more informed pruning decisions


Knowledge Distillation: Train a sparse student model with structured constraints to mimic a dense teacher model.


```python
# code of 2:4 sparsity, how did it derived

def apply_2_4_sparsity(weights):
	#Reshape to group of 4
	groups = weights.reshape(-1, 4)
	# Find 2 smallest magnitude weights
	indices = np.argsort(np.abs(group))[:2]
	# Zero them out
	group[indices] = 0
	return weights.reshape(original_shape)
```