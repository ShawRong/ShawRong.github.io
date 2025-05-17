# Kinds
- Agglomerative methods
	It begin from clusters based on each data point to a big cluster containing all the data point
- Divisive Methods
	From big cluster containing all the data point to cluster containing one data point.
# Dendrogram

![[Pasted image 20250516131649.png]]

# Distance
- Single Linkage: The nearest distance between two point in two cluster
- Complete Linkage: The distance between two cluster is given by the distance between their most distant members
- Group Average Linkage: The average of the distance between all pairs of records (pair-wise)
- Centroid Linkage: The distance is defined as the distance between the mean vector of two cluster
- Median Linkage: The mid-point of the original two cluster cetnres is used as the centre of the new combined group. To avoid the characteristic properties of the small one are lost.


# Bottom-up(Agglomerative)
![[Pasted image 20250516133344.png]]
**single linkage:**
- We find the shortest distance between two cluster-> 2.0 between 1 and 2.
- We use the closest distance as the new distance, etc.

# Divisive Approach
## Polythetic Approach
It uses all the attributes to do clustering
### Procedure
In the begining, we select each point in the whole cluster, to form a new cluster, and measure the distance between these two cluster.

We select the cluster with greatest distance as new cluster A. The cluster containing other data points is called cluster B.

After that, we select one point a (one by one) from B, and measure the distance between A, and B/a, and measure the delta of distance of A and B/a.

We select the one with the greatest delta to put into A.

We continue this, until all delta become negative.
### Note
*We can continue to do this on each sub group to get more cluster*

## Monothetic Apporach

### Chi-Square Measure

![[Pasted image 20250516143935.png]]
note: ad and bc is cross product.  a + b ... is row and column-wise addition.
### Do clustering
![[Pasted image 20250516144020.png]]
We use the chi-square to get the one with greatest impact.
For example, for A: $\chi_{AB}^2 + \chi_{AC}^2$
for B: $\chi_{AB}^2 + \chi_{BC}^2$
etc...

We select the most impact one, and cluster by the value. 
For example, the most impact one is attribute A. We split the whole group into {2, 3, 4} and {1, 5} due to they get same value 0 or 1.



