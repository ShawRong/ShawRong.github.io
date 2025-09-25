---
title: "FGMP notes"
date: 2025-09-25T09:49:26.189Z
draft: false
tags: []
---

Abstract
========
- FGMP is meaning "fine-grained mixed precision" quantization.
- This is a method that quantize weights and activations. The activations are quantized on the fly.
- Their co-design hardware help to achieve greater energy efficiency.
- First, They develop a policy using perturbation weighted by the Fisher information to select which weight and activation blocks to keep in higher precision.
- Their method is use some metric to select blocks, to keep them in higher precision.
- Second, they proposed a new clipping method to help low-precision blocks can be in good accuracy, too.
- They propose hardware augmentations, which encompasses 1) data path support at block granularity, 2) mixed-precision activation quantization unit. 
- Result: <1% perplexity degradation on Wikitext103 for llama 2 7b. 14 less energy, and 30% less weight memory, compared with fp8 baseline.

Summary
========

Related Work:
--------
**Hardware support for mixed precision quantization**
- GOBO is something store outlier in 32bit precision, and do quantization to other ones
- OLVE is using some encoding method, to store outlier values in high precision by sacrificing the neighboring dense values.
- MicroScopiQ retains outlier values by pruning out a portion of non-outlier values.
- SPARK uses encoding scheme to represent different magnitude values to different precision by using a mete data for each element.
- FGMP uses block level granularity instead of element level comparing with SPARK to save space
- FGMP is using efficient **vector** multiply accumulate. I think we can't use tensor core in this senario.

Question
--------
- [ ] In the description of their policy, they said that "it's a perturbation in each value and weighted by the Fisher Information." I can't fully understand its meaning. We need further reading.
- [ ] What's the hardware "augmentation"? Do they really devised their new hardware? Or it's just some simulation.
- [ ] What's the quantization method for the weight. What's the on-the-fly quantization method for activation?
- [ ]