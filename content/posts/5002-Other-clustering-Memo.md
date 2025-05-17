+++
date = '2025-05-17T21:17:29+08:00'
draft = false 
title = '5002 Other Clustering Memo'
+++
# Methods
- Model-Based Clustering (EM algorithm)
- Density-Based Clustering (DBSCAN)
- Scalable Clustering Method (BIRCH)

# EM algorithm
## why?
- previous method, each point belongs to a single cluster. No point belongs to different cluster with probabilities. 
## Normal distribution
$p(x|<\mu, \sigma>)=\frac{1}{\sqrt{2\pi}\sigma}e^{- \frac{(x-\mu)^2}{2\sigma^2}}$
sigma = standard derivation, mu = mean.
## Procedure
- Initialize all $\mu_i$ and $\sigma_i$ (random)
- For each point x, we calculate its probability belong to cluster i. (formula: $p(x\in C_i) = \frac{p(x|<\mu_i, \sigma_i>)} { \sum p(x|<\mu, \sigma>})$, i.e. probability of cluster i divided by the sum of probability of all the clusters)
- We calculate the new mean of cluster, and update $\mu$. using formula($\mu_i = \sum_{X} x \frac{p(x\in C_i)}{\sum_y p(y\in C_i)}$. (i.e. according to the probabilities that all points belong to cluster i)
Repeat until converge
# DBSCAN
## why?
traditional clustering cannot handle irregular shaped cluster
# Concepts
## $\epsilon$-neighborhood
denote by N(p), is the set of points that distance with p is within $\epsilon$ (including itself).
## Kinds of point
- core points, N(p) is at least MinPts(a parameter).
- border points, Not core point, but N(p) contains at least one core point
- noise points, not core or border
## procedure
- Each cluster contains at least one core point (create cluster from core point)
- If core point p and q, and N(p) contain q (or inverse), merge the cluster into one. (they are in the same cluster)
- Border point is assigned to previous created cluster if it contains the core point (arbitrarily if many).
- All noise points do not belong to any.

# BIRCH
## why?
most previous algorithm cannot handle update, and not scalable.
## Concepts
- Mean $\mu = \frac{\sum x}{n}$
- Radius $R = \sqrt{\frac{\sum (x-\mu)^2}{n}}$, average distance from member to the mean.
- Diameter $D=\sqrt{\frac{\sum\sum(x_i-x_j)^2}{n(n-1)}}$, average pair-wise distance within a cluster.
## Procedure
L = {}
when there is a new data point x comes in:
- If L is empty: create cluster C containing x, and insert into L.
- If not empty: Find closest cluster C of x, insert x into C. When C has diameter D greater than a given threshold, splite C into 2 sub-clusters, and insert them into L, remove the old one. (split: using dendrogram or other)
## Acceleration
- n: no. of points in the cluster
- LS: the linear sum of n points $\sum x_i$
- SS: the square sum of the n points $\sum x^2$
Update them is feasible and fast.
Comparison of time consumption of each update: O(n^2) and O(1). 