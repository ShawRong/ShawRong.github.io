+++
date = '2025-05-16T21:17:29+08:00'
draft = false 
title = '5002 Neural Network Memo'
+++
# Activation Function
## Threshold function, Step function, Hard Limiter
$$
y = \begin{cases}
1 \quad \text{if net } \geq 0\\
0 \quad \text{if net } < 0
\end{cases}
$$


## Linear Function
Just Identiy Function

# Rectifier Function / Rectified Linear Unit(ReLU)
$$
y = \begin{cases}
net \quad \text{if net } \geq 0\\
0 \quad \text{if net } < 0
\end{cases}
$$
# Sigmoid Function
![[Pasted image 20250515160346.png]]
# tanh Function
![[Pasted image 20250515160408.png]]
## Forward(single neuron)
$\text{net} = w_1x_1 + w_2x_2 + b$
$y = \frac{1}{1 + e^{-net}}$
## Error Backward Propagation
$w_1 = w_1 + \alpha(d - y)x_1$

*when we processed all data points in the whole dataset, we say that we processed the dataset in one epoch.*
