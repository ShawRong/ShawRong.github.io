+++
date = '2025-05-16T21:17:29+08:00'
draft = false 
title = '5002 Bayesian Classifier Memo'
+++
# Conditional Probability
$P(B | A) = \frac {P(A | B) P(B)} {P(A)}$
Assuming x, y, z are independent, we have:
$P(x, y, z | A) = P(x | A) + P(y | A) + P(z | A)$

# Naive Bayesian Classifier
Assume all the attribute are independent, and we get:
$$\begin{align}
P(yes | race, income) &= \frac {P(race, income | yes) P(yes)} {P(race, income)} \\
&= \frac {P(race | yes) P(income | yes) P(yes)} {P(race, income)} 
\end{align}$$


The $P(race, income)$ are identical when calculating no.

So we just calculate $P(race | yes) P(income | yes) P(yes)$


# Bayesian Belief Network

## Conditionally independent
$P(C | A, B) = P(x | B)$
We say C is conditionally independent of A given B.

Property used in Bayesian Belief Network: A node is conditionally independent of its non-descendants if its parents are known.

### Different Condition to Calculate
![[Pasted image 20250515143618.png]]

We know how to calculate P(HD|E), very simple bayesian rule.
$P(HD=yes|E=yes) = 0.25$
And
$P(E=yes|HD=yes)=\frac{P(HD=yes|E=yes)P(E=yes)}{P(HD=yes)}$
$P(E=yes) = 0.7$
$P(HD=yes) = P(HD=yes|case1)P(case1) + P(HD=yes|case2)P(case2) = 0.31$
And we can calculate it.

And there are 
$P(HD=yes|E=yes, BP=yes, CP=yes)$
We need to swap using formula $P(A|B, C) = \frac{P(B|A, C)P(A|C)}{P(B|C)}$.
$P = \frac{P(BP=yes, CP=yes | HD = yes, E=yes)P(HD=yes|E=yes)}{P(BP=yes, CP=yes|E=yes)}$
And another rule $P(A|B) = P(A|case1, B)P(case1|B) + P(A|case2, B)P(case2 | B)$ here used here:
$P(BP=yes, CP=yes|E=yes) = P(BP=yes, CP=yes|case1, E=yes)P(case1|E=yes) + P(BP=yes, CP=yes|case2, E=yes)P(case2|E=yes)$
$= P(BP=yes, CP=yes|HD=yes)P(HD=yes|E=yes) + P(BP=yes, CP=yes|HD=no)P(HD=no|E=yes)$

## Disadvantage of Bayesian Belief network
- The bayesian belief network classifier requires a predifined knowledge about network.
- The bayesian belief network classifier cannot work directly when the network contains cycles.






