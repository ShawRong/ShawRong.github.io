---
title: "Idea"
date: 2025-07-13T10:19:07.748Z
draft: false
tags: []
---

name: unstructured mixture quantization.

idea: 
consider a tile(8x8) of matrix,
for each row(token level), we set a constant sparsity level, such as 4/8. 
we keep 4 out of 8 to be fp16, the rest 4 to be fp8.
we get:
f16 stands for fp16, f8 stands for fp8.
\[f16, f16, f16, f16,  f8, f8, f8, f8\]
\[f16, f8, f16, f16,   f16, f8, f8, f8\]
\[f16, f16, f16, f16,  f8, f8, f8, f8\]
\[f8, f16, f16, f8,   f16, f8, f16, f8\]
... (8 row in total)

and we factorize fp16 into multiply result of fp8. and we get


\[2xf8, 2xf8, 2xf8, 2xf8,  f8, f8, f8, f8\]
\[2xf8, f8, 2xf8, 2xf8,   2xf8, f8, f8, f8\]
\[2xf8, 2xf8, 2xf8, 2xf8,  f8, f8, f8, f8\]
\[f8, 2xf8, 2xf8, f8,   2xf8, f8, 2xf8, f8\]
...

we get dimension from 4x8 -> 4x12. 
byte for each row changes from fp16 * 8 -> fp8 * 12 i.e. 16 byte -> 12 byte.

If we keep only one fp16, we get 16byte -> 9byte

and we need a bitmap, looks like:
\[1, 1, 1, 1, 0, 0, 0, 0\]
\[1, 0, 1, 1, 1, 0, 0, 0\]
\[1, 1, 1, 1, 0, 0, 0, 0\]
\[0, 1, 1, 0, 1, 0, 1, 0\]
...

or : \[0, 1, 2, 3, 8, 10, 11, 12, 16, 17, 18, 19, ...\]


It takes about 8byte for a tile using first way. or we need ...


we can generate a mask matrix from the first one, and we can do:

Assume we get previous matrix note as M, and a vector x, we get:
QK^T = X 

and we can separate K into two matrices such that K = K_1 + K_2

so, we have:
QK^T = Q(K_1 + K2)^T = QK_1^T + QK_2^T = X

suppose we have matrix M, s.t.
MK = K_1
M'K = K_2
K_1 + K_2 = K
(M + M') K = MK + M'K = K_1 + K_2 = K
M + M' = I
M' = I - M

we just need M for mask of fp16.

when doing calculation, we use mask matrix M, suppose the matrix of mixture precision, is K.

Q (MK)^T + Q ((I - M)K)^T

of course, in the 1st term, we need to convert 2xfp8 into fp16 during calculation. this is trivial.

It's like:

\[2xf8, 2xf8, 2xf8, 2xf8,  f8, f8, f8, f8\]
\[2xf8, f8, 2xf8, 2xf8,   2xf8, f8, f8, f8\]
\[2xf8, 2xf8, 2xf8, 2xf8,  f8, f8, f8, f8\]
\[f8, 2xf8, 2xf8, f8,   2xf8, f8, 2xf8, f8\]
...

after masking:

\[2xf8, 2xf8, 2xf8, 2xf8,  0, 0, 0, 0\]
\[2xf8, 0, 2xf8, 2xf8,   2xf8, 0, 0, 0\]
\[2xf8, 2xf8, 2xf8, 2xf8,  0, 0, 0, 0\]
\[0, 2xf8, 2xf8, 0,   2xf8, 0, 2xf8, 0\]
...

convert to fp16.

\[fp16, fp16, fp16, fp16,  0, 0, 0, 0\]
\[fp16, 0, fp16, fp16,   fp16, 0, 0, 0\]
\[fp16, fp16, fp16, fp16,  0, 0, 0, 0\]
\[0, fp16, fp16, 0,   fp16, 0, fp16, 0\]
...

multiply

and we do mask to get fp8:

\[0, 0, 0, 0,  f8, f8, f8, f8\]
\[0, f8, 0, 0,   0, f8, f8, f8\]
\[0, 0, 0, 0,  f8, f8, f8, f8\]
\[f8, 0, 0, f8,   0, f8, 0, f8\]
...

do the calculation.

after this, we add to get the correct anwser ...

then we can use this for mixture precision