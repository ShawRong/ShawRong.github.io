---
title: "Nestquant Note"
date: 2025-09-25T09:49:36.585Z
draft: false
tags: []
---

Abstract
========
- This is a PTQ scheme for weights and activations that based on self-similar nested lattices.
- It's said NestQuant have mathematically shown to be information theoretically optimal.
- The authors implement a practical low-complexity version of NestQuant base on Gosset Lattice.


Questions
========
- [ ] What's self-similar nested lattices? What's the purpose of this design. Why it's helpful for quantization.
- [ ] It's said that recent works have mathematically shown such quantizers to be "information theoretically optimal". Is is shown in this paper, or another paper?
	The paper called 'Gosset lattice spherical vector quantization with low complexity' is the source to prove self-similar nested lattices to be information-theoretically optimal.
- [ ] What's Gosset Lattice. Why it's chosen to be the practical way to implement this so-called NestQuant?
	Gosset lattice is E8 lattice.
- [ ] What's the using method of storing these quantized vector of weight and activations or KV-cache? Is it still vector quantization?
- [ ] What's QA-LDLQ, what's the change comparing with the original LDLQ?
- [ ] The author of NestQuant believes that the normal quantization method, diving by the L-inf norm to get entires to be in \[-1, 1\] is suboptimal. There are two reasons, one is that uniform quantization induces error that is distributed uniformly on the small cube.
	- The author uses L2 norm to normalize these data, so that these data point are not get into a unit sphere. $\hat{x} = x/\|x\|_2$ .
- [ ] How does QA-LDLQ works, can we adopt it to a mxfp 4 version?
	- It's needed to add a noise to the original activation to model the quantized activation. Here authors use gaussian distribution to model the noise. 
	- Let W to denote the original weight matrix, and U is the quantized weight matrix. Therefore, the original proxy loss becomes $\sigma(U) := WX - U(X + Z)$, $E[\|\sigma(U)\|^2]$. 

Summary
========
- In the related work section of LLM quantization, it state that the method including QuIP# and QTIP which uses residual vector quantization and trelis codebook appear to be too expensive to apply in runtime. That perhaps explaining why non-uniform quantization for activations and KV-cache was not attempted before this work. This part is trying to explain that we cannot use this method similarly to apply on activation and KV-cache.
- It state previous work LDLQ as "replacing the MSE distortion loss to be a weighted MSE loss dependent on the statistics of the incoming activations", following the authors of QuIP, QuIP# and GPTQ.
- However, when we extend this method to activation quantization, it's natural to understand that the statistic of activation itself will definitely change comparing with the original activations. So the authors modified the LDLQ, which is called QA-LDLQ.
- The author hold a belief that previous work employ normalization to handle Gaussian-like data. The data were divided by the L_inf norm to get entries to be in \[-1, 1 \], which forms a unit cube. And the shape is not suitable for the sphere-like gaussian distribution. To solve these problem, author use normalization by the L2 norm and points inside the Voronoi region of a Gosset lattice.




Todo
========
- [ ] Learn what's nested lattices.
- [ ] Read the paper called 'Gosset lattice spherical vector quantization with low complexity'
- [ ] Conway&Sloane "Sphere Packings, Lattices and Groups"(Chapter 1)
- [ ] Micciancio & Goldwasser "Complexity of Lattice Problems"

Summary
========