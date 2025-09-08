---
title: "QuIP sharpe Note"
date: 2025-09-08T11:49:22.555Z
draft: false
tags: []
---

Abstraction
========
There are 3 improvement based on the previous work QuIP. 
- It uses Hadamard transform instead of previous random matrix.
- It uses vector quantization to take advantage of the ball-shaped sub-Gaussian distribution, specifically introducing a set of hardware-efficient codebooks based on the highly symmetric E_8 lattice.
- It uses fine-tuning to improve fidelity to the original model.

Notation
========
- The only important thing for me right now, is the hadamard matrix part.
- The vector quantization for sub-Gaussian distribution is not that fast as we can see.
- Finetune is not what I want to use, which can be complicate
- There it involves a blockLDLQ, which should pay attention to.



Summary of Hadamard Matrix Rotation
========

Question
--------
- [x] What's the detail to prove that the theoretical bound on the incoherence parameter is improved by using RHT
	It states that "QuIP#'s RHT achieves superior incoherence via a log dependence on the matrix size rather than the Kronecker method's log-squared dependence".
- [x] What's the log dependence and log-square dependence?
	It gives a Lemma, to provide us the incoherence parameter, to show that this parameter is of a formula of n with a log dependence, which can be superior compared with the parameter analyzed in the previous paper QuIP.
- [ ] What's the random? How to generate a RH?
	It states that "In the previous paper, the incoherence processing is performed by conjugate weight matrix W and proxy hessian H by structured random orthogonal matrices. QuIP# uses the RHT, which performs x to VSx where V is Hadamard matrix, and S is a random sign vector {+- 1}^n".
- [x] I still don't understand how to perform conjugate by using x to VSx, where is conjugation?
	It says after this statement clarifying the procedure of performing rotation is like "VSHSV^T" and "USWSV^T".
- [ ] What's the U here? What's the S_U here? Is it ok to use different conjugation part in the front and the end? I think you need to check the original paper.

Summary
--------
- It replace the original 2-factor Kronecker product by a Randomized Hadamard Transformation(RHT)
- The original Kronecker product is from the previous paper "QuIP". This Kronecker product is involved due to the difficulty and large scale of the random matrix. And we can solve the problem by using Kronecker product which can accelerate the inference.
- By using RHT, there are 3 improvement. First, the theoretical bound on incoherence parameter is improved. Second, the asymptotic cost of multiplying by structured orthogonal matrix is improved. Third, the cost to multiply is further reduced. It's show by experiment that the perplexity is improved by performing this change.
- There are constraint to construct any dimension hadamard matrix. So the author proposed Randomized Fast Fourier Transform (RFFT) with similar runtime and concentration properties as the RHT. RFFT requires dimension to be even, which is a weaker constraint compare with the RHT.

Summary of BlockLDLQ
========

Question
--------
- [ ] What's central limit theorem? Is it "RHT transformed weight follow a roughly ball-shaped Gaussian distribution?"
- [ ] 


Summary
--------
- It states that "It follows from the central limit theorem that RHT-transformed weights follow a roughly ball-shaped Gaussian distribution"

Adaptive Rounding
========

Summary
--------