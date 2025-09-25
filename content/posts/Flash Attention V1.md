---
title: "Flash Attention V1"
date: 2025-09-25T09:49:07.358Z
draft: false
tags: []
---

Abstract Summary
========
- Time and memory complexity of Self-Attention is quadratic in sequence length.
- Attention Algorithm is IO-aware, a arguement made by the authors
- They are using tiling to reduce the number of memory read/write. Therefore, less HBM access is required.
- Additionally, they extend FlashAttention to block-sparse attention
- 3x speedup on GPT2...

Question
========
- [ ] What's better-than-chance
- [ ] what's block-sparse atttention
- [ ] make self-attetention definition more clear.
- [ ] what's the clear definition of wall-clock?
- [ ] what's IO-aware meaning, need to make it clear.
- [ ] 

Summary
========
Introduction
--------
- We know tiling can help us to calculate a tile of the result without loading entire matrix. And we can use this calculated tile of result to multiply and make an furture result. this is what FlashAttention doing. Each little tile of the product of K and V is furture multiply with the Q, to produce the new tile of the final result sm(QK)V. Obviously, there we have a sm() function to solve, to furture use this tile.(Tiling, and new problem how to solve sm() in a tile)
- I think the way the draw the graph of FlashAttention calculation is great, especially the arrow to point out which is outer loop and which is inner loop.
- Apparently, they did kernel fusion.

Related Work
--------
- I think there is a very intereting and important thing is about the dimesion of the these matrices, Q, K, and V should be in dimmension of Nxd. Therefore we can produce a Nxb matrix as result. In the computation midterm, Q and K produce a matrix in dimension of dxd.
- I think there is a pretty good image to describe this Algorithm 0, which is original Attention Implemention. You can check Bili videos.



Method
--------
- They are using two technology: **Tiling* & **Recomputation**
- Their primitive target is to reduce the amount of HBM accesses.
- This is the way the authors describe their softmax:"By scaling the output of each block by the right **normalization** factor before adding them up, we get the correct result at the end."
- We know the original softmax can be calculated with decomposition like following formula.
 $$
 \begin{align*}
m(x) &:= \text{max}_i (x_i) \\
f(x) &:= [e^{x_1 - m(x)}, \cdots,e^{x_B - m(x)}], \quad \text{B stand for the len of vector} \\
\mathcal{l}(x)&:= \sum_i f(x)_i\\
\text{softmax}(x)&:=\frac{f(x)}{\mathcal{l}(x)}
 \end{align*}
 $$
And we can reformulate this by: 
$$
\begin{align*}
m(x) &= m [x^{(1)}, x^{(2)}] \\
f(x) &= \left[e^{m(x^{(1)}) - m(x)} f(x^{(1)}), e^{m(x^{(2)}) - m(x)} f(x^{(1)})\right] \\
\end{align*}
$$
Here,
$$
\begin{align*}
e^{m(x^{(1)}) - m(x)} f(x^{(1)}) &= e^{m(x^{(1)}) - m(x)} [e^{x_1 - m{(x^{(1)})}}, \cdots,e^{x_B - m{(x^{(1)})}}]  \\
&= [e^{x_1 - m{(x)}}, \cdots,e^{x_B - m{(x)}}]  \\
\end{align*}
$$
And
$$
\begin{align*}
\mathcal{l}(x) = \mathcal{l} ()...
\end{align*}
$$
You can check the original paper for the formula, I don't want to type anymore, this keyboard is hard to use.

- We can see from this new formula that we can first calculate the first f1 and f2. and the l1 and l2. After that we can calculate the final mx, and use this final mx to compute the final softmax. 

Question
--------
- [ ] But there is a serious problem is that if we want to calculate the tile, we have to load a whole row and col. So we need to see if they really load a whole row or col.
- [ ] Why we need a m(x) in each decomposed softmax compuation? Can we use a random number? We should dive into details to know more.