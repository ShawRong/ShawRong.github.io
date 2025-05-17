Authority weight (in-degree)
Hub weight (out-degree)

- A good authority has many edges from good hubs
- A good hub has many edges from good authorities

# Hit Algorithm
Two steps:
- Sampling Step
- Iteration Step


## sampling step
Given a user query with several terms, we collect a set of pages that are very relevant -- called the base set.

- We get all the web page that contain any query terms. This is called root set.
- We find the link pages, which are linked by or link to web pages of root set.
- Base set = link pages & root set

## Iteration Step
We know the hub weight of a web page is given by:
$h(N) = a(N) + a(MS) + a(A)$
etc..
so we can get:
$\vec{h} = [{h(N), h(MS), h(A)}]$
$\vec{a} = [{a(N), a(MS), a(A)}]$
$\vec{h} = M\vec{a}$
Here M represents adjacent matrix.

Similarly, we get 
$\vec{a} = M^T\vec{h}$
So we get:
$\vec{h} = MM^T\vec{h}$
$\vec{a} = M^TM\vec{a}$


### How to interation?
- We set all the pages with hub weight = 1 at first.
- And we do  $\vec{h} = MM^T\vec{h}$, to get new vec h.
- And we **normalize** it to make the sum of all the element in the vector to be equal to the number of elements.  For example, normalize to 3.



# Recommendation
- We can rank base on h weight
- We can rank base on a weight
- We can rank base on a combination of h & a weight (for example, sum)


# Pagerank Algorithm
# Comparison with HIT algorithm
- Disadvantage of hit: It can be hard to determine to use a or h.
- Advantage of Pagerank: only one concept for ranking

# Ideas
Stochastic Matrix
- column-wise Matrix, column is decided by the out-link of a node. And the sum of the column elements equal to 1.

Like: 
N -> N
N -> A 
A -> N 
A -> M
M -> A

We can get:
![[Pasted image 20250515235758.png]]

# Iteration
After we get the stochastic matrix
- Initial the vec r to be [1, ..., 1].
- $\vec{r} = M \vec{r}$
- so on so forth
We need to understand that it's actually the weighted sum of columns, so the sum of elements equal to the num of elements.
![[Pasted image 20250516000110.png]]


# Spider Trap
Definition: A group of one or more pages that have no links out of the group will eventually accumulate all the importance of the web.

**How to avoid it?**
$\vec{r} = 0.8W\vec{r} + \vec{c}$
$\vec{c} = \begin{bmatrix} 0.2 \\ 0.2 \\ 0.2\end{bmatrix}$
