---
title: "Quantization Calculation"
date: 2025-07-23T02:26:08.672Z
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