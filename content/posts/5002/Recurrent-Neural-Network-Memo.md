+++
date = '2025-05-16T21:17:29+08:00'
draft = false 
title = '5002 Recurrent Neural Network Memo'
+++
# Difference with Neural Network
The output of RNN are passed to the next RNN network (internal variable).

# Multilayer
Actually, these rnn can have multiple layers and multiple memory units.
# Basic RNN
The basic rnn is simple. It use internal variable(s) as output variable(y). And the s can be calculated by using:
$s_t = tanh(W [x_{t}, s_{t-1}] + b)$
$y_t = s_t$


# LSTM
There are components:
- internal variable to store memory
- forget feature to forget some portion of internal variable
- input feature to decide portion of input and strength of input 
- output feature to decide portion of output and strength of output
## Some significant difference
First, the output and internal state previous are input, too.


## Components and Framework
Component:
- Forget gate (portion of memory)
- Input gate (portion of input)
- Input activatin gate (weight of input)
- New internal state is Input(Input gate combine with Input activation gate + Forget gate combine with previous Internal state)
- Output gate (portion of output)
- Final output state gate (Multiplication of tanh internal state and Output gate)
Formulas:
- Forget gate$f_t = \sigma(W_f [x_t, y_{t-1}] + b_f)$
- Input gate $I_t = \sigma(W_i (x_t, y_{t-1})+ b_i)$
- Input Activation gate $a_t = tahn(W_a[x_t, y_{t-1}]+b_a)$
- New internal state: $s_t = I_t \times a_t + s_{t-1}$
- Output gate: $O_t = \sigma(W_o [x_t, y_{t-1}] + b_o)$
- Final output: $O_t \times \tanh(s_t)$


# Gated Recurrent Unit
## Advantages
- training time shorter due to simple architecture
- few data point to capture properties
- no interval variable
## Key difference
No internal varible here, just use previous prediction y.
## Component
- Reset Gate: using previous prediction as reference to store memory. Portion of memory
- Input activation Gate: just input activation gate
- Output: Combine portion of predicted target and portion of the processed input variable. (ratio come from the update feature)
### Summary
- reset component
- input activation component
- update component
- final output component
## Formula
- reset gate: $r_t = \sigma(W_r [x_t, y_{t-1}] + b_r)$
- input activation gate: $a_t = \tanh(W_a [x_t, r_t \times y_{t-1}] + b_a)$
- update gate: $u_t = \sigma(W_u [x_t, y_{t-1}] + b_u)$
- final output: $y_t = (1-u_t) y_{t-1} + u_t a_t$
