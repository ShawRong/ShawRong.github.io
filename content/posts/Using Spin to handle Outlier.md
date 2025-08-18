---
title: "Using Spin to handle Outlier"
date: 2025-08-06T06:24:19.137Z
draft: false
tags: []
---

# Adaptive Rounding
Basically, it's a new method based on the motivation that nearest rounding might not be the best approach for quantization rounding.

## Motivation: Round to Nearest is Not that Good.
If me perturb the weights of a pre-trained model, it's easy to find the expectation of the loss difference shows like this:
$$
\mathbb{E}[\mathcal{L}(\mathbf{x}, \mathbf{y}, \mathbf{w} + \Delta \mathbf{w}) - \mathcal{L}(\mathbf{x}, \mathbf{y}, \mathbf{w})] \approx \frac{1}{2} \Delta \mathbf{w}^T \mathbf{H}^{(\mathbf{w})} \Delta \mathbf{w} 
$$
Here, we use Taylor expansion, and omit the gradient part due to we assume network is trained to convergence leading to ignoring gradient term.
$$
\Delta \mathbf{w}^T \mathbf{g}^{(w)} = 0
$$
Here, authors give us an example to explain why round to nearest is sub-optimal.

**Example 1.**: Assume $$\Delta \mathbf{w}^T = [\Delta w_1, \Delta w_2]$$ and 
$$
\mathbf{H}^{(w)} = \begin{bmatrix} 1 &0.5 \\ 0.5 & 1\end{bmatrix}
$$
apparently, here 
$$
\Delta \mathbf{w}^T \mathbf{H}^{(\mathbf{w})} \Delta \mathbf{w} = \Delta w_1^2 + \Delta w_2^2 + \Delta w_1 \Delta w_2
$$
For the diagonal entries, i.e. $\Delta w_1^2$ or w_2, we know it's smaller when we use round to nearest. But for the off-diagonal entries, we  opposite signs of the two perturbations improve the loss. **To minimize the overall impact of quantization on the task loss, we need trade-off between the contribution of the diagonal terms and the off-diagonal terms**

> **Experiment**: the authors made a experiment to prove the sub-optimal of round-to-nearest. They generate 100 stochastic rounding and apply it to the 1st layer of ResNet18, and found 48 stochastically sampled rounding choices lead to a better performance than rounding-to-nearest. And they also see that rounding all values **up, or all down**, has an **catastrophic** effect.

## Task Loss & Rounding
**per-layer optimization**
Obviously, the weight after quantization get only two choice. one is ceil and one is floor (choosing different rounding method).
So it's a binary optimization problem, following this target:
$$
\text{arg} \min_{\Delta \mathbf{w}} \mathbb{E}[\mathcal{L}(\mathbf{x}, \mathbf{y}, \mathbf{w} + \Delta \mathbf{w}) - \mathcal{L}(\mathbf{x}, \mathbf{y}, \mathbf{w})]
$$
If we want to cost, we need a forward pass, for every $\Delta \mathbf{w}$ setup. 
So here, the author (or most researcher), utilize the second order **Taylor series approximation**, and ignore the interactions among weights belonging to different **layers**.

> **Quoting:** Additionally, we ignore the interactions among weights belonging to different layers.
> This, in turn, implies that we assume a **block** diagonal $\mathbf{H}^{(\mathbf{w})}$(here, it's like a diagonal-block-like matrix), where **each non-zero** block corresponds to one layer.

the **per-layer** optimization problem looks like:
$$
\text{arg} \min_{\Delta \mathbf{w}^{(l)}} \mathbb{E}[\mathbf{g}^{(\mathbf{w}^{(l)})} \Delta \mathbf{w}^{(l)} + \frac{1}{2}\Delta \mathbf{w}^{(l)^T}\mathbf{H}^{(\mathbf{w}^{(l)})^T}\Delta \mathbf{w}^{(l)}]
$$
we can ignore the gradient term, and we can see the optimization problem becomes:
$$
\text{arg} \min_{\Delta \mathbf{w}^{(l)}} \mathbb{E}[\Delta \mathbf{w}^{(l)^T}\mathbf{H}^{(\mathbf{w}^{(l)})^T}\Delta \mathbf{w}^{(l)}] \quad (e)
$$
the difference is the Hessian matrix is for the **layer-wise** weight compare with previous approximation.

> **Experiment**: The authors plot a picture to show the relationship between accuracy and proxy(the quadratic hessian term).

Here the author suggest two problem about solving this **proxy for optimization**. 
1. H is hard to compute(computational and memory complexity).
2. Solving this optimization problem is NP-hard. 

----
**1. The complexity associated with $H^{(\mathbf{w}^{(l)})}$**
This is from article 'Practical Gauss-Newton Optimization for Deep Learning'.

**sample parameter hessian**, *H* which we discuss earlier, has element likes:

$$
[H]_{ij} = \frac {\partial^2}{\partial \theta_i \partial \theta_j} E(\theta)
$$
Here, E stand for the loss. 
The expected parameter Hessian is similarly given by the expectation of this equation.

**Comment:**
> **parameter hessian vs. sample hessian**: parameter hessian is the second derivatives of the objective function. sample hessian is the second derivatives with respect to individual training samples or datapoint. one is partial to weight w, and one is partial to data point feature x.
> what's really about the gradient and hessian? When we compute gradients, we are asking "how does the loss change with respect to parameters for this specific data point/batch?" Different data points will generally produce different answers to this question. So as to Hessian. When we compute the Hessian of a loss function, we are asking "How does the curvature of the loss surface change with respect to parameters for this specific data point/batch?"
> So basically, different data points create different hessians. Some inputs might create steep, narrow valleys in the loss landscape, while others create gentle, broad curves. A data point where the model makes a large error might contribute high curvature, while a point where the model performs well might contribute relatively flat curvature.

the partial derivative of loss with respect to the weights of layer lambda is given by:

*Note: a, b here stand for a index of the weight.*
$$
\frac{\partial E}{\partial W_{a, b}^{\lambda}} = \sum_{i} \frac{\partial h_i^{\lambda}}{\partial W_{a, b}^{\lambda}} \frac{\partial E}{\partial h_i^{\lambda}} = a_b^{\lambda - 1} \frac{\partial E}{\partial h_a^{\lambda}} \quad (1)
$$
where:
$$
h_a^{\lambda} = \sum_{b} W_{a, b}^{\lambda} \cdot a_b^{\lambda - 1} \quad (2)
$$
**According to (2), for i eq a, we get**:
$$
\frac {\partial h_a^{\lambda}}{\partial W_{a,b}^{\lambda}} = a_b^{\lambda - 1}
$$
**and for i neq a, we get 0.**
$$
\frac {\partial h_i^{\lambda}}{\partial W_{a,b}^{\lambda}} = 0
$$
so, the equation (1) holds.

so for the hessian of layer lambda, we have:
$$
[H_{\lambda}]_{(a, b), (c, d)} = \frac{\partial^2 E}{\partial W_{a,b}^{\lambda}\partial W_{c,d}^{\lambda}}
$$
$$
\frac{\partial}{\partial W_{c,d}^{\lambda}}(a_b^{\lambda - 1} \frac{\partial E}{\partial h_a^{\lambda}}) = a_b^{\lambda - 1} \sum_i \frac{h_i^{\lambda}}{\partial W_{c,d}^{\lambda}} \frac{\partial}{h_i^{\lambda}}(\frac{\partial E}{\partial h_a^{\lambda}})
$$

$$
\frac {\partial h_c^{\lambda}}{\partial W_{c,d}^{\lambda}} = a_d^{\lambda - 1}
$$
and for i neq c, we get 0.
$$
\frac{\partial}{\partial W_{c,d}^{\lambda}}(a_b^{\lambda - 1} \frac{\partial E}{\partial h_a^{\lambda}}) = a_b^{\lambda - 1} \sum_i \frac{h_i^{\lambda}}{\partial W_{c,d}^{\lambda}} \frac{\partial}{h_i^{\lambda}}(\frac{\partial E}{\partial h_a^{\lambda}}) = a_b^{\lambda - 1} a_d^{\lambda - 1} \frac{\partial^2 E}{\partial h_a^{\lambda} \partial h_b^{\lambda}} =  a_b^{\lambda - 1} a_d^{\lambda - 1} [\mathcal{H}_{\lambda}]_{a,c}
$$

**In conclusion,**
$$
[H_{\lambda}]_{(a, b), (c, d)} = a_b^{\lambda - 1} a_d^{\lambda - 1} [\mathcal{H}_{\lambda}]_{a,c}
$$
$$
[\mathcal{H}]_{a,c} = \frac{\partial^2 E}{\partial h_a^{\lambda} \partial h_b^{\lambda}} 
$$
**Note**: here we use curlycue H stand for the so called *pre-activation Hessian*. and the normal H stand　for sample Hessian.

$$
H_{\lambda} = \frac{\partial^2 E}{\partial \text{vec} (W_{\lambda}) \partial \text{vec} (W_{\lambda})} = (a_{\lambda - 1} a_{\lambda - 1}^T) \otimes \mathcal{H}_{\lambda}
$$
where otimes stands for Kronecker product.

**Back to where we begin from the paper Ada Rounding.**
Indeed, it's quite hard to compute the $\mathcal{H}_{\lambda}$, which is the hessian of loss with respect to activation of layer lambda. 

To tackle this, we make **assumption** that the Hessian of the task loss w.r.t the pre-activations, i.e. $\mathcal{H}_{\lambda}$ is a **diagonal** matrix. 

In the article, it looks like:
$$
\mathbf{H}^{\mathbf{w}^{(l)}} = \mathbb{E}[\mathbf{x}^{(l-1)}\mathbf{x}^{(l-1)^T} \otimes \text{diag}(\nabla_{\mathbf{z}^{(l)}}^2 \mathcal{L}_{i,i})]
$$
Considering the previous optimization problem, it becomes, (considering every single w_k).

Kronecker factorization property:
$$
(\mathbf{A}\otimes\mathbf{B})\text{vec}(\mathbf{X}) = \text{vec}(\mathbf{B}\mathbf{X}\mathbf{A}^T)
$$
$$
\text{vec}(\mathbf{X})^T (\mathbf{A} \otimes \mathbf{B}) \text{vec}(\mathbf{X}) = \text{tr}(\mathbf{X}^T \mathbf{B}\mathbf{X}\mathbf{A}^T)
$$
'vec' here is vectorize the matrix column by column.
here, 
$$
\Delta \mathbf{w}_k^{(l)} = \text{vec}(\Delta \mathbf{W}_{k,:}^{(l)})
$$
so, to optimization problem becomes:
$$
\text{arg} \min_{\Delta \mathbf{W}_{k,:}^{(l)}} \mathbb{E}[\nabla_{\mathbf{z^{(l)}}} \mathcal{L}_{k,k} \cdot \Delta \mathbf{W}_{k:}^{(l)}\mathbf{x}^{(l-1)}\mathbf{x}^{(l-1)^T} \Delta \mathbf{W}_k^{(l)^T}]
$$
> **Quoting:** The problem is a per-layer problem, now.

There authors made an **assumption**, that the gradient of Loss is a **constant independent of the input data samples.**
The optimization problem becomes:
$$
\text{arg} \min_{\Delta \mathbf{W}_{k,:}^{(l)}} \mathbb{E} [(\Delta \mathbf{W}_{k,:}^{(l)}\mathbf{x}^{(l-1)})^2] \quad (1)
$$
or 

$$
\text{arg} \min_{\Delta \mathbf{W}_{k,:}^{(l)}} \Delta \mathbf{W}_{k,:}^{(l)}\mathbb{E} [\mathbf{x}^{(l-1)}\mathbf{x}^{(l-1)^T}]\Delta \mathbf{W}_{k,:}^{(l)^T} \quad (2)
$$
> **Quoting:** This optimization requires no knowledge of the subsequent layers and the task loss. We are just simply minimizing the Mean Squared Error(MSE) introduced in the pre-activations z due to quantization. 
> **Comment**: Here x is the activation of this layer. i.e. h^(l) = W^(l) x^(l-1) 

> **Quoting**: The optimization problem in (1) can be tackled by either precomputing the second moment, as in (2), and then perform the optimization over perturbation of weight. Or by performing a single layer forward pass for each potential perturbation of weight, during the optimization procedure.

> **Comment:** here author said, 'In section 5, we empirically verify that the constant constant diagonal approximation of (the gradient^2 of loss) term does not negatively influence the performance'. they did experiment using equation (e) and equation (2) to check the performance. And authors claim it's ok to make these assumption, since it seems works pretty well.

*We skip the Ada round part to start with QuIP.*
# QuIP

Short Introduction
----
The authors in this paper introduce previous work including **Adaptive rounding** and **GPTQ**.

**Adaptive Rounding**: The authors uses the sota **proxy objective**, a optimization problem described earlier in this passage. 
**GPTQ**: this is a novel rounding method that can work on largest language model(OPT and BLOOM at that time).

About the proxy objective of the quantization, we have this:
$$
\mathcal{l}(\hat{W}) = \mathbf{E}_x [\|(\hat{W} - W)x\|^2] = \text{tr}((\hat{W} - W)H(\hat{W} - W)^T).
$$
We can easily see the crucial part of this objective is the so-called proxy hessian(i.e. second moment matrix), which should be calculated while backward propagating or pre-calculated.

> **Quoting:** Crucially, this formulation lets the quantization be run in parallel across neurons, which is tractable for large language models. 

Derivation of the 'Optimal' Adaptive Rounding Method
----

The author define sequences of adaptive rounding methods with respect for optimizing this objective.

the feature of the methods is **column** by column based on the basis that this proxy objective is a **per-layer** proxy objective.

> **Quoting:** This scheme rounds columns one at a time; at each step, it adds a 'correction' term that is a linear function of the residual from the rounding we have done so far(previous column).

The formula of each quantized weight matrix is:
$$
\hat{W_k} = \mathcal{Q}(W_k + (W_{1:(k-1)} - {\hat{W}_{1:(k-1)}})a_k)
$$
here, W_k with hat stands for k-th column of quantized weight matrix(a **vector**). Q stands for quantization of each elements of vector. W with index like 1:(k-1), is selected weight matrix with selected columns (not selected as zero, to keep shape uniform). And a_k here stand for the linear function of the residue.

And finally, the final quantized matrix **satisfies**:
$$
\hat{W} = \mathcal{Q}(W + (W - \hat{W})U) \quad (1)
$$
here, U is a upper-triangular matrix, consisting of the vector a.

This is a definition of a family of **adaptive rounding** based on features including "column by column".

here the authors denote quantization error of the **Adaptive rounding** (not simple Q(W) - W, since here we are adaptive rounding):
$$
\eta = \mathcal{Q}(W + (W - \hat{W})U) - (W + (W - \hat{W})U)
$$
replace the Q term with equation (1)
$$
\eta = (\hat{W} - W) + (\hat{W} - W)U
$$
$$
\eta = (\hat{W} - W)(I + U)
$$
Therefore, we have the previous proxy objective becomes:
$$
\text{tr}((\hat{W} - W) H (\hat{W} - W)^T) = \text{tr}(\eta(U+I)^{-1} H (U + I)^{-T}\eta^T)
$$
And If we choose decomposition of H to becomes:
$$
H = (U' + I)D(U' + I)^T \quad (2)
$$
And we assign U (a linear function of residual) to be U', we will achieve:
$$
\text{tr}(\eta H \eta^T)
$$
> **Quoting**: We denote as LDLQ (LDL decomposition) the rounding procedure in Eq.(1) with U to be U' as the LDL assignment from Eq.(2). We **will now see** that the LDL assignment of U is fact optimal.

**Deriving the Optimality of the LDLQ**
----
It defines a worst and average loss of each quantization methods, and show the Loss of LDLQ satisfies the following:
$$
\frac{m}{4} \text{tr}(D) = \mathcal{L}_{\text{worst}}(\text{LDLQ}, H) \leq \mathcal{L}_{\text{worst}}(\mathcal{A}, H)
$$
$$
\frac{m}{c} \text{tr}(D) = \mathcal{L}_{\text{avg}}(LDLQ, H) \leq \mathcal{L}_\text{avg} (\mathcal{A}, H)
$$
here m stands for the number of rows being quantized. 
mathcal A stands for any quantization method following Eq.(1).

The authors states: 
> **Quoting**: LDLQ achieves strictly lower worst and average-case proxy loss because $$\text{tr}(D) < \text{tr}(\tilde{H})$$tilde H stands for non-diagonal H. 

And somehow: 
$$
\mathcal{L}_{\text{worst}}(\text{LDLQ}, \tilde{H}) < \mathcal{L}_{\text{worst}} (Stoch, \tilde{H})
$$
and
$$
\mathcal{L}_{\text{worst}}(\text{LDLQ}, \tilde{H}) < \mathcal{L}_{\text{worst}} (Stoch Or Nearest, \tilde{H})
$$
The authors use 2 lemma and 1 definition to prove this.

first the author define what's \mu incoherence. (\mu stands for the level of incoherence).
And derive an inequality of tr(D) and tr(H^{1/2})^2

> **Quoting:** To the best of our knowledge, this is a novel result using incoherence to obtain a bound on tr(D) that depends only on the spectrum of H.

And author use this to show these two inequality holds for the LDLQ.

> **Quoting:** This shows that for sufficiently low-rank H, LDLQ is asymptotically better than plain nearest and stochastic rounding by a factor $\mu^2k/n$

The need for Incoherence
-----
The authors were able to show LDLQ gets **asymptotially** better bound in terms of just the spectrum of H. 
**Was the incoherence assumption necessary to get this result**?
**Yes.**
**Without incoherence: no improvement with a spectral bound**.