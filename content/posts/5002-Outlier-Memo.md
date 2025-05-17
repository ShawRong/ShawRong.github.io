+++
date = '2025-05-16T21:17:29+08:00'
draft = false 
title = '5002 Outlier Memo'
+++
# Comparison
Statistical Model Disadv: Assume that the data follows a particular distribution.
Distance-based Model Adv: No assume distribution. But not density wise.
Density-Based Model Adv: can find local outliers
# Concept
$\epsilon$ is the distance between p and the k-th nearest neighbor.
local reachability density lrd_k(p) is: $\frac 1 \epsilon$
local outlier factor(LOF) is $(\sum_{o \in N_k(p)} \frac{lrd_k(o)}{lrd_k(p)})/{k}$
N_k(p) is the $\epsilon$-neighborhood of p (excluding p).
