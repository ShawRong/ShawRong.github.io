# Linear Support Vector Machine
We can generalize this problem as this:
$$
\begin{align}
\min w_1^2 + w_2^2 \\
\text{subject to } y(w_1x_1+w_2x_2+b) \geq 1
\end{align}
$$
## Why minimizing this term?
Because we want to maximize the distance D between two line. The D is involved in w by $\frac{w}{D}$.

## Why transform maximizing to minimizing?
We want to transform the objective function from a non-linear form to a quadratic form.  
Then, the problem becomes a form of quadratic programming which has many existing efficient techniques for that.

### Non-linear one
- step1: transform the data into a higher dimensional space using a 'nonlinear' mapping
- step2: use the linear support vector machine in this high-dimensional space

