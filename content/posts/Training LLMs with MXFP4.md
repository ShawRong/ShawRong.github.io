---
title: "Training LLMs with MXFP4"
date: 2025-09-25T09:49:30.022Z
draft: false
tags: []
---

# Abstract Summary
- Using MXFP 4 GEMMs, which are 2x faster than FP 8 on supported hardware. Because the training is mostly computation bound, at the forward and backward stage.
- The key insight is to compute unbiased gradient estimates with stochastic rounding.
- Directly applying SR to MXFP 4 can result in high variance from block-level outliers. 
- Their recipe computes > 1/2 the training FLOPs in MXFP4, in the computation aspect. And it enables an estimated speedup of >1.3 x over FP8, >1.7x over BF 16 during back propagation.

Summary
========
- LLM training is compute bound in matrix multiplication. LP GEMMs can accelerate training.
- It's shown in section 4 that directly using MXFP4 in even only the backward pass of decoder linear layers significantly degrades model quality.
- Their method hinges on computing low-variance, unbiased gradient estimates.
- Their method is: first use stochastic rounding to compute unbiased GEMMs. Then, they use a memory bound construction of random Hadamard transform to reduce the effect of outliers and theoretically bound the variance of SR.
- Gap between MXFP4 and BF16 is < 0.1 validation perplexity on pre train GPT model.

IEEE 754
--------
- IEEE 754 format: $(-1)^S (1 + M) 2^{E - \text{bias}}$ .
- This exponent mantissa construction means FP datatypes are scale invariant with respect to quantization signal-to-noise ratio(SNR) bar over/underflow. It means SNR is not variant with the scale when using this FP format without considering over/underflow.
- While the exact training setup may differ, the core bottlenecks of training are the compute-bound forward and backward passes that calculate the loss and gradient, respectively.

MP
--------
- In MP(Mixed Precision), parameters are kept in high precision and GEMM operands are converted to a LP datatype for a LP GEMM. Quantization usually has minimal overhead.
- However, quantization introduces distortion in the GEMM operands. And since the forward and backward passes all happen in low precision, both the loss and the model updates can deviate from their true values.
- There it gives an example that FP8 MP recipes typically use E4M3 in the forward pass and E5M2 in the backward pass due to the different properties of gradient, weights and activations..

Stochastic Rounding
--------
- NR(nearest rounding) method rounds each high precision number to its closest representable value in the LP datatype. But NR is not unbiased, which can be detrimental to low precision training.
- SR(stochastic rounding) can achieve unbiased rounding, which randomly rounds a number to a representable value in the LP datatype so that the rounded number equals the original number in expectation.
- SR can be implemented through dithering. It adds random uniform noise to the input number and then performs NR to achieve a unbiased rounding, which randomly rounds a number to a representable value in the LP datatype.
- The dithering for a uniform integer and non-uniform one is different, which requires modifying the noise scale. The dithering looks like: 
- $$\begin{align} \delta &= \mathcal{U}(-0.5, -0.5) \quad (1)\\ \text{SR}_\text{dither} (x) &= \begin{cases}\lfloor x \rfloor \quad x + \delta < \lfloor x \rfloor + \frac 1 2 \\ \lceil x \rceil \quad x + \delta \geq \lfloor x \rfloor + \frac 1 2 \end{cases} \quad (2) \end{align}$$

MXFP4 Quantization Process
--------

MXFP4 is a block-based floating-point format, the block size usually to be 32. 32 elements shared a same exponent number with format E8, which represents the all 8 bits are for exponent. Each mxfp4 is consist of 1 signal bit and 3 mantissa bits.

**Format Structure (Per Block)**:
- **Grouping:** Data is processed in blocks of 32 values.
- **Shared Exponent:** One 8-bit exponent is shared across the entire block.
- **Data Values:** Each value is a 4-bit signed integer (1 sign bit + 3 magnitude bits), representing numbers from **-6 to +6**.
- **Total Storage:** 136 bits per block (8 bits for the exponent + 32 * 4 bits for the values).

**Quantization Algorithm:**
- **Find Block Maximum:** For a block of 32 numbers, find the one with the largest absolute value (`max_abs_val`). This value sets the scale for the entire block. Our goal is to map the original numbers to be in the range of \[-6, 6\].
- **Calculate Shared Exponent:** We know if we want to achieve this range. We get a inequality: (max_abs_val / scale) <= emax. Here, it should be like: max_bas_val / (2^shared_exp) <= 6. To get the shared_exp, we can use this fomula: shared_exp = ceil(log2(max_abs_val / 6))
- **Scale & Quantize:** Create a scaling factor from the exponent (`Scale = 2^shared_exp`). Divide every number in the block by this `Scale` and round it to the nearest integer.
- **Store:** The final stored data consists of the single 8-bit `shared_exp` and the 32 new 4-bit integer values.

The original mxfp 4 quantization algorithm:

```python
"""
Convert vector of scalar float in high precision to an mx block {e8, 32 fp4}.
"""

# exponent of the largest normal in lower precision type
# here it should be 2.
emax = 2
max_abs_per_block = torch.max(torch.abs(weight_blocks), dim=1)[0]
shared_exp = floor(torch.log2(max_abs_per_block)) - emax.unsqueeze(1)

normalized = weigth_blocks / (2^shared_exp).unsqueeze(1)
# Quantize to FP4 indices, according to its closest value indices
quantized_indices = self._quantize_to_fp4_indices(normalized)

# Pack indices to bytes
packed_blocks = self._pack_indices_to_bytes(quantized_indices)

return packed_blocks, scales
```

The optimized version of mxfp4 quantization:


```python
"""
Optimized Version
Convert vector of scalar float in high precision to an mx block {e8, 32 fp4}.
"""

# exponent of the largest normal in lower precision type
# here it should be 2.
emax = 2
max_abs_per_block = torch.max(torch.abs(weight_blocks), dim=1)[0]
shared_exp = floor(torch.log2(max_abs_per_block)) - emax.unsqueeze(1)

# Do a uniform scaling to the weight blocks 
# make it scale to 3/4 original.
weight_blocks = weight_blocks * 3.0 / 4.0
normalized = weigth_blocks / (2^shared_exp).unsqueeze(1)
# Quantize to FP4 indices, but we use stochastic rounding to FP4.
quantized_indices = self._quantize_to_fp4_indices_stochastic(normalized)

# Pack indices to bytes
packed_blocks = self._pack_indices_to_bytes(quantized_indices)

return packed_blocks, scales
```
Questions
========
- [ ] Why directly applying SR to MXFP 4 can result in high variance form block-level outliers?
- [ ] It says that MXFP4 uses an INT8 scale s for every contiguous block v of 32 FP4 numbers to represent 2^{s-1}v. where 1 is the exponent bias for FP4. I got a question for what's the bias here? Is this universal? 
- [ ] What's the stochastic rounding here. And how can it used to compute the GEMMs? And how does the Hadamard transform applied? It's applied to what?
- [ ] What if we just use the Hadamard transform? Is there any ablation study?
- [ ] What does they do to avoid the overhead of the RHT and SR.
- [ ] IEEE 754 format: $(-1)^S (1 + M) 2^{E - \text{bias}}$ . Does fp4 obey this format?
- [ ] Here it says about the back propagate through a linear layer, we need to calculate the gradient of y with respect to x. Why? Shouldn't we calculate the gradient like Loss to W? And it's says something like: dL/dx = dL/dy W, dL/dW = dL/dY x, and dL/db = 1 dL/ dy. What's all this, and what's our purpose? 
- [ ] Why the dL/dx and dL/dW is computationally intensive?
- [ ] How can this dithering achieve a random rounding with its expectation to be equal with the original one?
- [ ] It says: "For example, near the end of the training, the model update norm is much smaller than the parameter norm and information in low precision updates can be "lost". Here, stochastic rounding can be used to preserve the update in expectation". I don't really understand why keep the update in expectation can help the training near the end.