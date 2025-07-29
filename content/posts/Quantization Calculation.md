---
title: "Quantization Calculation"
date: 2025-07-23T15:08:37.198Z
draft: false
tags: []
---

# Symmetric Quantization
suppose we get data in range \[r_min, r_max\].

and suppose we get 'b' bits to quant.

we can get scale by using r_max, only
```python
b = 4 #suppose int4
quant = (2^(b-1)) - 1 # how many positive number in int4
scale = r_max / quant

x = 10
x' = round(x / scale)
```

$$
\Delta = \frac {\text{max}}{2^{b - 1} - 1}
$$
$$
x' = \text{round}(\frac{x}{\Delta})
$$
$$
x = x'\Delta
$$
# Asymmetric Quantization

we need r_min and r_max to calculate the scale, and we need to shift zero point to the middle to make:
```python
scale * (q_min - zero_point) = r_min
r_min / scale = q_min - zero_point
zero_point = q_min - r_min / scale
#we know, q_min = -2^(b-1)
zero_point = -2^(b-1) - round(r_min / scale)
```

$$
\Delta = \frac{\text{max} - \text{min}}{2^{b} - 1 }
$$
$$
z = -2^{b-1} + \text{round}(\frac {\text{min}}{\Delta})
$$
$$
x' = \text{round}(\frac{x}{\Delta} + z)
$$
$$
x = \Delta * (x' - z)
$$


# GGUF Quant.
GGUF(Group-wise Quantization).
divide matrix into **block**, and we do quant. to individual block. each block are equipped with a scale.

default: group in **row**, each 32 elements is a group .


# GPTQ

## Optimal Brain Surgeon
for a object function (loss function), L, we can do taylor expansion.
$$
L(\mathbf{w}) = L(\mathbf{w}_0) + g^T(\mathbf{w} - \mathbf{w}_0) + \frac{1}{2}(\mathbf{w} - \mathbf{w}_0)^T\mathbf{H}(\mathbf{w} - \mathbf{w}_0) + o(\|\mathbf{w} - \mathbf{w}_0\|^3)
$$
here g stand for gradient and H stand for hessian matrix.

we can assume g is 0, since we achieve a local minimum.

and we denote w - w_0 as \delta w
$$
\Delta \mathbf{w} = \mathbf{w} - \mathbf{w}_0
$$
we can get, delta L is:
$$
\Delta L = \frac{1}{2} \Delta \mathbf{w}^T \mathbf{H} \Delta \mathbf{w}
$$
suppose we update, w_i to 0, which is pruning, we can get a formula:
$$
\Delta w_i + w_i = 0
$$
which is: (e_i stand for one-hot vector at index i)
$$
\mathbf{e}_i^T \Delta \mathbf{w} + w_i = 0
$$
we can get our optimization problem with constraint, at each step when we want to prune w_i. (use lagrange).

$$
\min_{\Delta \mathbf{w}, i} \frac{1}{2} \Delta \mathbf{w}^T \mathbf{H} \Delta \mathbf{w} + \lambda(\mathbf{e}_i^T\Delta\mathbf{w} + w_i)
$$
$$
\frac{\partial}{\partial \Delta \mathbf{w}} \left[ \frac{1}{2} \Delta\mathbf{w}^T \mathbf{H} \Delta\mathbf{w} + \lambda (\mathbf{e}_i^T \Delta \mathbf{w} + w_i) \right] = 0

$$
gives
$$
\mathbf{H} \Delta \mathbf{w} + \lambda \mathbf{e}_i = 0
$$
$$
\Delta \mathbf{w}^* = -\lambda \mathbf{H}^{-1} \mathbf{e}_i
$$
gives
$$
\frac {\partial L}{\partial \lambda} = \mathbf{e}_i \Delta \mathbf{w} + w_i = 0
$$
i.e.
$$
\lambda^* = \frac{w_i}{(\mathbf{H}^{-1})_{ii}}
$$
$$
\Delta \mathbf{w} = - \frac{w_i}{(\mathbf{H}^{-1})_{ii}} \mathbf{H}^{-1}\mathbf{e}_i
$$
so, we should update after each prune, like:
w_i <- w_i + \Delta w^*

## Gradient Parallel Surgeon
suppose, we want:
here, E is a matrix where each column is a diagonal matrix, with 1 or 0, to keep w_i or make w_i to be 0. And we have constraint like:
$$
\Delta w_i + w_i = 0
$$
$$
\mathbf{E}^T (\Delta \mathbf{w} + \mathbf{w}) = \mathbf{0}
$$
Therefore, we get our lagrange:
$$
L(\Delta \mathbf{w}, \mathbf{\lambda}) = \frac{1}{2} \Delta \mathbf{w}^T \mathbf{H} \Delta \mathbf{w} + \mathbf{\lambda}^T \mathbf{E}^T(\Delta \mathbf{w} + \mathbf{w})
$$
$$
\frac{\partial L}{\partial \Delta \mathbf{w}} =  \mathbf{H} \Delta \mathbf{w} + \mathbf{E} \mathbf{\lambda} = 0
$$
$$
\frac{\partial L}{\partial \mathbf{\lambda}} = \mathbf{E}^T(\Delta \mathbf{w} + \mathbf{w}) = 0
$$
$$
\Delta \mathbf{w}^* = - \mathbf{H}^{-1} \mathbf{E}(\mathbf{E}^T \mathbf{H}^{-1}\mathbf{E})^{-1}\mathbf{w}
$$
**solution 1**: use the diagonal approximation:
$$
\text{diag}(\mathbf{H}) = (\frac{\partial ^2 L}{\partial w_0^2}, ...)
$$
calculation can be very quick.