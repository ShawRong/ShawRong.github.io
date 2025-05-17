# Methods
- Dense Unit-based Method
- Entropy-Based Method
- Transformation-Based Method
## why subspace clustering?
dimension curse: when the number of dimensions increases, the distance between any two points is nearly the same.
# Dense Unit-based Method
## Property
If a set is cluster in k dimension, then it's part of cluster in k-1 dimension, too.

## Algorithm
- Identify sub-space that contain dense units (aprior algorithm, find x, y, z these dimension)
- Identify clusters in each sub-space that contain dense units (base on previous found x, y, z dimension, we find cluster like {x:\[1, 10\] y: \[11, 20\]}, etc..)

## adapt aprior
for example, we get grid of 10, and mark $X_1$ as \[1, 10\], and so on. threshold 1.
( 1, 19) $X_0$ $Y_1$
(11, 29) $X_1$ $Y_2$
(21, 39) $X_2$ $Y_3$
(31, 49) $X_3$ $Y_4$
(41, 59) $X_4$ $Y_5$

We get dense unit:
$L_1 = \{X_0, X_1, X_2, X_3, X_4\}$
$L_1 = \{Y_1, Y_2, Y_3, Y_4, Y_5\}$
We found subspace:
$L_1 = \{\{X\}, \{Y\}\}$
And
$C_2 = \{\{X, Y\}\}$
For attribute $\{X, Y\}$, we get:
$C_2 = \{\{X_0, Y_1\}, \{X_0, Y_2\} \cdots\}$
$L_2 = ..$
And ...


# Entropy-Based Method
## conditional entropy
H represents entropy.
$H(Y|X)=\sum_{x\in A} p(x)H(Y|X=x)$
$H(Y|X=x)= -\sum_{y \in B} p(y|X=x)\log(p(y|X=x))$
And
$H(X, Y) = H(X) + H(Y|X)$
$H(X_1, X_2, \cdots, X_k) = H(X_1,\cdots, X_{k-1}) + H(X_k|X_1,\cdots, X_{k-1})$

So we can get the Lemma:
If subspace with k dimension has good clustering (H(X1,..., X_k) <= w).
The each of the (k-1) dimensional projections has good clustering

## Apriori
We can calculate:
H(Age)=0.12, H(Income)=0.08, w = 0.2
- L_1 = {{Age}, {Income}}
- C_2 = {{Age, Income}}... etc

H(Age,...) is calculated based on grid.

## comparison with dense unit base
good to find subspace, only. bad to find clusters, only.
# Transformation-based Method
## steps
for example, data points (3, 3)(0, 2)(-1, -1)(2, 0).
- calculate the mean vector of data points (1, 1)
- Get the difference vector with mean vector(2, 2)(-1, 1)(-2, -2)(1, -1)
- Get the covariance Matrix 1. (arrange data vector as column).
	$$Y = \begin{bmatrix}2 &-1 &-2 &1 \\ 2 &1 &-2 &-1\end{bmatrix}$$
- 2. covariance matrix: $\Sigma = 1/4 YY^T$
- Find eigenvalues of eigenvectors, solve $det(\Sigma - \lambda I)=0$. (cross product), get lambda
- Find eigenvectors using eigenvalues, $(\Sigma - \lambda_1 I)x = 0$
- Normalize the eigen vector into standard form.
- Arrange the eigen vector into transformation matrix. (a column a vector, corresponding to the eigenvalue).
- Using $y = T x$ to transform the data vector.
- Only keep the row corresponding to smallest eigen value of the transformed data vector.

## why smallest eigen value? 
information loss is minimized if we project to the eigenvector with a large eigenvalue.
