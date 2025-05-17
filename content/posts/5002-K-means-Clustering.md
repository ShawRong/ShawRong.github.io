+++
date = '2025-05-16T21:17:29+08:00'
draft = false 
title = '5002 K-means Clustering Memo'
+++
# K-means
First, we get a pre-defined parameter k.
We make k random guesses of the mean point.
Do iterations until the mean doesn't change.
- Assign each data point to the cluster whose mean is nearest
- Calculate the mean of each cluster
- Replace the cluster with a new mean
## About the initialization of k-means
The popular way to initialize the starting mean is a random choice.
The result depends on the initial guess, and a suboptimal result is possible, so we do several different starting points.
## Disadvantages of k-means
- Bad initial guess can lead to no points assigned to the cluster.
- The value k is not user-friendly, cause we don't know the number of clusters in the beginning.
# Sequential k-means
To deal with data that comes at different times. We choose to update our mean point when a new data point is encountered.
## Procedure
We set several initial guess points.
We save variable n to represent the number of data points in the cluster(no mean point).
Do iterations until interrupted:
- Get the next data point
- Find the closest mean point
- Increment the number counter n
- Replace the mean point by $m+(1/n)(x - m)$ (m for mean point and x for data point, n for counter)
## How does this formula work
When data point t comes in, we get:
$m_t = m_{t-1} + \frac 1 t (x - m_{t-1})$
Suppose $m_{t-1} = \sum_{1}^{t-1} x_i / (t - 1)$,
We get :$m_{t-1} (t - 1) = \sum_{1}^{t-1} x_i$
$m_{t-1} (t - 1) + x_t = \sum_{1}^{t} x_i$
$m_{t-1} (t - 1)/t + x_t/t = m_t$
$m_t = m_{t-1}+ 1/t( x_t - m_{t-1})$
# Forgetful Sequential K-means
## Formula
$m_n = (1-a)^n m_0 + a\sum (1-a)^{n-k}x_k$
can be derived as:
$m_{i+1} = m_i + a(x-m_i)$
a ranges from 1 to 0.
