---
title: "GPTQ Note"
date: 2025-09-25T09:49:32.874Z
draft: false
tags: []
---

Abstract
========


GPTQ Method Summary
========

Questions
--------
- [ ] It said that "As a consequence, the set of unquantized weights F and similarly H_F^{-1} is always the same for all rows." I don't quite understand it. 
- [ ] What' the procedure of this quantization specifically? I mean what's the Inverse Layer Hessain, and weight matrix / block. I think I can see that the inverse layer hessian seems to follow a row major way, and the weight matrix follows a column major way? 
- [ ] In the background part, it mentions that they assume that the quantization grid of W is fixed before the process, and that individual weights can move freely as in. I don't quite understand this.
- [ ] It still need to read more things about the hessian thing. First question definitely to be why hessian looks like this. Second, why there are full-precision weights involved?
- [ ] I found it's hard to read these formula. What's the qq and q here?
- [ ] More numerically stable than direct matrix inversion if we use Cholesky Decomposition. But why?
- [ ] Why small eigenvalues often correspond to noise rather than signal, and why it says directions with little statistical support get massive influence?  



Summary
--------
- It found that the order does not matter when do the quantization one by one in the pretty large language model. So it draw a insight that any fixed order may perform well, especially on large models.
- 


Dampening Part
--------
In GPTQ quantization, when a weight is quantized, remaining weights must be updated to compensate for the quantization error. 

And from the perspective of Matrix, we get matrix conditioning theory to help.

--------
**Matrix conditioning theory**: 
For any invertible matrix A:
$$
\kappa(A) = \|A\|\|A^{-1}\|
$$
This is the definition of the condition number of an invertible matrix A.

And since we know:
$$
\begin{align}
\|A\| &= \sigma_{max}(A) \\
\|A^{-1}\| &= \sigma_{max}(A^{-1}) = 1/\sigma_{min}(A) \\
\kappa(A) &= \|A\|\|A^{-1}\| = \sigma_{max}(A) / \sigma_{min} (A)
\end{align}
$$

Alternative form: 
$$
\kappa (A) = \lambda_{max}(A) / \lambda_{min} (A) = \sigma_{max}(A) / \sigma_{min}(A)
$$
Since for symmetric positive definite matrices, we have:
$$
\begin{align}
A^T A &= A^2 \\
\lambda_i(A^2) &= \lambda_i(A)^2 \\
A^TAv &= A^T\lambda v = \lambda Av = \lambda^2 v \\
\sigma_i(A) &= \sqrt{\lambda_i(A^2)} = \sqrt{\lambda_i(A)^2} = \lambda_i(A)
\end{align}
$$

For a system Ax = b, with a error perturbation $\delta x$, we can see:
$$
\|\delta x\| / \|x\| \leq \kappa(A) \|\delta b\| / \|b\|
$$
**This mean that condition number bounds relative error amplification.**

--------
**Ill-conditioning problem**
This comes to a definition of ill-conditioning problem.
- Large condition number $\kappa(A)$
- Small eigenvalues, even some $\lambda_i = 0$
- Near-singular: $det(A) = 0$

We don't have to explain the condition number thing, since large condition number means large bound of relative error amplification of a system Ax=b.

So why small eigenvalues cause huge updates?
Idk, need more explaination.

--------
**Dampening**

$H_{damped} = H + \lambda I$

Here lambda is the 1% of average diagonal value of H.

By using dampening, we can shift the original eigenvalue from $\lambda_1, \lambda_2, \cdots, \lambda_n$ to $\lambda_1 + \lambda, \lambda_2 + \lambda, \cdots, \lambda_n + \lambda$.
Therefore, the
$$
\begin{align}
\kappa(H) = \lambda_{max} / \lambda_{min} \approx \lambda_{max}/0 = \infty \\
\kappa(H + \lambda I) = (\lambda_{max} + \lambda) / (\lambda_{min} + \lambda)
\end{align}
$$
Therefore, we can prevent the problem leading by the small eigenvalue.

--------
GPTQ uses **Cholesky** for numerical stability.

$H = LL^T$, where L is lower triangular.

The advantage of the Cholesky Decomposition is:
- More numerically stable than direct matrix inversion.
- Efficient for positive definite matrices
- Well-optimized kernel

When we invert a matrix A directly, the condition number of $A^{-1}$ is same as A, but the errors get amplified. With Cholesky decomposition, we are working with triangular matrices L that typically have better condition numbers than the original matrix A.

Direct matrix inversion using methods like Gaussian elimination involves many division operation and intermediate results that can accumulate floating-point errors. 

In conclusion, using Cholesky decomposition to solve inversion can lead to less error compared with the Gaussian elimination.

--------
In conclusion, the GPTQ algorithm optimize to be like:
$H^{-1} = \text{Cholesky}((2XX^T + \lambda I)^{-1})$