---
title: "Training LLMs with MXFP4"
date: 2025-09-08T11:48:28.689Z
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